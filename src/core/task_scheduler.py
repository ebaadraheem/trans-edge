"""
src/core/task_scheduler.py
--------------------------
Transfer-Aware Node Selection (TANS) — the algorithmic "Delta" over AMP4EC.

Background
----------
AMP4EC's NSA scores nodes on:
  Total_Score = 0.2·S_R + 0.2·S_L + 0.1·S_P + 0.5·S_B              (Eq. 7)

CarbonEdge adds a carbon efficiency score S_C:
  Total_Score = wR·S_R + wL·S_L + wP·S_P + wB·S_B + wC·S_C          (Eq. 3)

The "Delta" in this implementation — Transfer-Aware Node Selection (TANS) —
extends the carbon score to account for *both* compute and data-transfer
emissions:

  Total_Carbon = (E_comp × CI_local) + (E_trans × CI_network)

  SC_TANS = 1 / (1 + Total_Carbon)

where:
  E_comp  = P_node × T_avg / 3_600_000         (kWh, compute energy)
  E_trans = size_mb / 1024 × coeff_network      (kWh, link transfer energy)
  CI_local / CI_network = gCO₂/kWh for the regional grid

Scheduling Modes (Table I, CarbonEdge)
---------------------------------------
  Performance  : wC = 0.05  – prioritise fast nodes
  Balanced     : wC = 0.30  – compromise
  Green        : wC = 0.50  – minimise carbon
  TANS-Green   : wC = 0.60  – transfer-aware green mode (novel)

Node skipping rules (Algorithm 1, AMP4EC)
------------------------------------------
  • load  > 0.8  → overloaded, skip
  • latency > threshold → too slow, skip
  • insufficient resources → skip
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

from src.core.carbon_monitor import (
    TRANSFER_ENERGY_KWH_PER_GB,
    CarbonMonitor,
    NodeCarbonState,
)
from src.core.model_partitioner import NodeProfile, Partition

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Scheduling mode weight configurations
# ---------------------------------------------------------------------------

class SchedulingMode(str, Enum):
    PERFORMANCE = "performance"
    BALANCED    = "balanced"
    GREEN       = "green"
    TANS_GREEN  = "tans_green"   # Novel: transfer-aware green


@dataclass(frozen=True)
class ModeWeights:
    wR: float   # resource availability
    wL: float   # current load
    wP: float   # historical performance
    wB: float   # balance / fairness
    wC: float   # carbon efficiency (TANS-extended)


MODE_WEIGHTS: Dict[SchedulingMode, ModeWeights] = {
    SchedulingMode.PERFORMANCE: ModeWeights(0.25, 0.25, 0.30, 0.15, 0.05),
    SchedulingMode.BALANCED:    ModeWeights(0.20, 0.20, 0.15, 0.15, 0.30),
    SchedulingMode.GREEN:       ModeWeights(0.15, 0.15, 0.10, 0.10, 0.50),
    SchedulingMode.TANS_GREEN:  ModeWeights(0.10, 0.10, 0.10, 0.10, 0.60),
}

# ---------------------------------------------------------------------------
# Live node state (load, task count, latency)
# ---------------------------------------------------------------------------

@dataclass
class LiveNodeState:
    """Runtime view of a single node – updated after each scheduling decision."""
    node_name:    str
    current_load: float = 0.0          # fraction 0–1
    task_count:   int   = 0            # active tasks
    net_latency_ms: float = 5.0        # last measured round-trip latency
    exec_time_history: List[float] = field(default_factory=list)

    MAX_HISTORY = 20

    def record_completion(self, exec_time_ms: float) -> None:
        self.exec_time_history.append(exec_time_ms)
        if len(self.exec_time_history) > self.MAX_HISTORY:
            self.exec_time_history.pop(0)
        self.task_count = max(0, self.task_count - 1)
        # heuristic load decay
        self.current_load = max(0.0, self.current_load - 0.15)

    def avg_exec_time(self) -> float:
        if not self.exec_time_history:
            return 200.0   # cold-start prior
        return sum(self.exec_time_history) / len(self.exec_time_history)

    def mark_scheduled(self) -> None:
        self.task_count   += 1
        self.current_load  = min(1.0, self.current_load + 0.15)


# ---------------------------------------------------------------------------
# Score components  (normalised to [0, 1])
# ---------------------------------------------------------------------------

def _score_resource(node: NodeProfile, live: LiveNodeState,
                    cpu_req: float = 0.1, mem_req_gb: float = 0.05) -> float:
    """
    S_R = (CPU_avail/CPU_req + MEM_avail/MEM_req) / 2    clamped to [0,1]
    """
    cpu_avail = max(0.0, node.cpu_cores * (1.0 - live.current_load))
    mem_avail = max(0.0, node.ram_gb    * (1.0 - live.current_load))
    r_cpu = min(cpu_avail / cpu_req, 1.0) if cpu_req > 0 else 1.0
    r_mem = min(mem_avail / mem_req_gb, 1.0) if mem_req_gb > 0 else 1.0
    return (r_cpu + r_mem) / 2.0


def _score_load(live: LiveNodeState) -> float:
    """S_L = 1 − current_load"""
    return 1.0 - live.current_load


def _score_performance(live: LiveNodeState) -> float:
    """S_P = 1 / (1 + avg_exec_time)   — smaller time → higher score"""
    return 1.0 / (1.0 + live.avg_exec_time())


def _score_balance(live: LiveNodeState) -> float:
    """S_B = 1 / (1 + task_count × 2)"""
    return 1.0 / (1.0 + live.task_count * 2)


def _score_carbon_tans(
    node: NodeProfile,
    carbon_state: NodeCarbonState,
    transfer_size_mb: float,
) -> float:
    """
    TANS carbon score — accounts for both compute and transfer emissions.

    Total_Carbon = E_comp × CI_local + E_trans × CI_network

    S_C = 1 / (1 + Total_Carbon)

    Both E values in kWh, CI in gCO₂/kWh → Total_Carbon in gCO₂.
    """
    # Compute energy estimate
    e_comp = carbon_state.estimate_inference_energy()   # kWh

    # Transfer energy estimate
    size_gb  = transfer_size_mb / 1024.0
    coeff    = TRANSFER_ENERGY_KWH_PER_GB.get(
        node.network_type, TRANSFER_ENERGY_KWH_PER_GB["default"]
    )
    e_trans  = size_gb * coeff                          # kWh

    ci = carbon_state.carbon_intensity_gco2_kwh         # gCO₂/kWh

    total_carbon = (e_comp + e_trans) * ci              # gCO₂
    return 1.0 / (1.0 + total_carbon)


# ---------------------------------------------------------------------------
# Scheduling result
# ---------------------------------------------------------------------------

@dataclass
class SchedulingDecision:
    selected_node: str
    partition_id:  int
    total_score:   float
    component_scores: Dict[str, float]
    estimated_carbon_gco2: float
    mode: str

    def __repr__(self) -> str:
        return (
            f"Decision(node={self.selected_node!r}, "
            f"partition={self.partition_id}, "
            f"score={self.total_score:.4f}, "
            f"carbon={self.estimated_carbon_gco2*1000:.4f} mgCO₂)"
        )


# ---------------------------------------------------------------------------
# TANS Scheduler
# ---------------------------------------------------------------------------

class TANSScheduler:
    """
    Transfer-Aware Node Selection Scheduler.
    """

    LATENCY_THRESHOLD_MS = 100.0   # skip nodes slower than this
    LOAD_THRESHOLD       = 0.80    # skip nodes with load > 80 %

    def __init__(
        self,
        nodes: List[NodeProfile],
        monitor: CarbonMonitor,
        mode: SchedulingMode = SchedulingMode.TANS_GREEN,
    ) -> None:
        self._nodes   = {n.name: n for n in nodes}
        self._monitor = monitor
        self._mode    = mode
        self._weights = MODE_WEIGHTS[mode]

        # Initialise live state for every node
        self._live: Dict[str, LiveNodeState] = {
            n.name: LiveNodeState(node_name=n.name) for n in nodes
        }

        self._scheduling_overhead_ms: List[float] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def mode(self) -> SchedulingMode:
        return self._mode

    @mode.setter
    def mode(self, new_mode: SchedulingMode) -> None:
        self._mode    = new_mode
        self._weights = MODE_WEIGHTS[new_mode]
        log.info("[tans] Switched to mode: %s", new_mode.value)

    def schedule(
        self,
        partitions: List[Partition],
        cpu_req: float = 0.1,
        mem_req_gb: float = 0.05,
    ) -> List[SchedulingDecision]:
        decisions: List[SchedulingDecision] = []
        t_start = time.perf_counter()

        prev_node_name = None  # FIX 3: Track the sender node

        for idx, part in enumerate(partitions):
            if idx == 0:
                incoming_mb = 0.0
            else:
                incoming_mb = partitions[idx - 1].output_tensor_size_mb

            decision = self._select_node(
                partition=part,
                incoming_transfer_mb=incoming_mb,
                prev_node_name=prev_node_name, # Pass it to the selector
                cpu_req=cpu_req,
                mem_req_gb=mem_req_gb,
            )
            decisions.append(decision)
            self._live[decision.selected_node].mark_scheduled()
            prev_node_name = decision.selected_node # Update for next iteration

        elapsed_ms = (time.perf_counter() - t_start) * 1000
        self._scheduling_overhead_ms.append(elapsed_ms)
        log.debug("[tans] Scheduled %d partitions in %.3f ms", len(partitions), elapsed_ms)
        return decisions

    def record_completion(
        self,
        node_name: str,
        exec_time_ms: float,
        partition_id: int = -1,
    ) -> None:
        """Call after a partition finishes executing on *node_name*."""
        self._live[node_name].record_completion(exec_time_ms)
        self._monitor.record(node_name, exec_time_ms)
        log.debug("[tans] Completion: node=%s part=%d t=%.1f ms",
                  node_name, partition_id, exec_time_ms)

    def avg_scheduling_overhead_ms(self) -> float:
        if not self._scheduling_overhead_ms:
            return 0.0
        return sum(self._scheduling_overhead_ms) / len(self._scheduling_overhead_ms)

    def full_snapshot(self) -> Dict:
        return {
            "mode":                   self._mode.value,
            "avg_overhead_ms":        round(self.avg_scheduling_overhead_ms(), 4),
            "decisions_made":         sum(
                                          len(l.exec_time_history)
                                          for l in self._live.values()),
            "cluster":                self._monitor.cluster_snapshot(),
            "live_nodes":             [
                                          {
                                              "node":         ls.node_name,
                                              "load":         round(ls.current_load, 3),
                                              "task_count":   ls.task_count,
                                              "avg_exec_ms":  round(ls.avg_exec_time(), 2),
                                          }
                                          for ls in self._live.values()
                                      ],
        }

    # ------------------------------------------------------------------
    # Internal: node scoring
    # ------------------------------------------------------------------

    def _select_node(
        self,
        partition: Partition,
        incoming_transfer_mb: float,
        prev_node_name: Optional[str],
        cpu_req: float,
        mem_req_gb: float,
    ) -> SchedulingDecision:

        w = self._weights
        best_score = -math.inf
        best_node  = None
        best_comps: Dict[str, float] = {}
        best_carbon = 0.0

        # FIX: Initialize the dictionary to hold candidate nodes before the loop
        eligible_nodes = {}

        for node_name, node in self._nodes.items():
            live   = self._live[node_name]
            carbon = self._monitor.nodes[node_name]

            # (Scheduler side): If co-located, incoming transfer is 0
            actual_transfer_mb = incoming_transfer_mb if node_name != prev_node_name else 0.0

            # Transfer relies on the SENDER'S network (the previous node)
            if prev_node_name and actual_transfer_mb > 0:
                prev_node = self._nodes[prev_node_name]
                net_type = getattr(prev_node, 'network_type', 'default').lower()
                sender_ci = self._monitor.nodes[prev_node_name].carbon_intensity_gco2_kwh
            else:
                net_type = getattr(node, 'network_type', 'default').lower()
                sender_ci = carbon.carbon_intensity_gco2_kwh
            
            # Align perfectly with the Execution Engine's math
            mbps = 100.0 if net_type in ["5g", "4g_lte"] else 500.0
            transfer_delay_ms = ((actual_transfer_mb * 8.0) / mbps) * 1000.0

            # Only filter based on base node health
            if live.current_load > self.LOAD_THRESHOLD or live.net_latency_ms > self.LATENCY_THRESHOLD_MS:
                continue

            # Resource sufficiency check
            cpu_avail = node.cpu_cores * (1.0 - live.current_load)
            mem_avail = node.ram_gb    * (1.0 - live.current_load)
            if cpu_avail < cpu_req or mem_avail < mem_req_gb:
                continue

            # Calculate base scores
            sR = _score_resource(node, live, cpu_req, mem_req_gb)
            sL = _score_load(live)
            
            sP=_score_performance(live) 
            if self._mode == SchedulingMode.TANS_GREEN:
                sP = 1.0 / (1.0 + (live.avg_exec_time() + transfer_delay_ms) / 100.0)
                 
            sB = _score_balance(live)
            
            # Calculate carbon metrics for this node
            e_comp = carbon.estimate_inference_energy()
            energy_per_gb = TRANSFER_ENERGY_KWH_PER_GB.get(net_type, TRANSFER_ENERGY_KWH_PER_GB["default"])
            
            # All modes calculate true actual transfer carbon
            e_trans = (actual_transfer_mb / 1024.0) * energy_per_gb
            compute_only_carbon = e_comp * carbon.carbon_intensity_gco2_kwh

            # full transfer-aware carbon (for TANS_GREEN)  
            total_carbon_eval   = compute_only_carbon + (e_trans * sender_ci)
            
            
            # Store metrics for Min-Max normalization later
            eligible_nodes[node_name] = {
                "sR": sR, "sL": sL, "sP": sP, "sB": sB,
                "raw_carbon": total_carbon_eval,     
                "compute_only_carbon": compute_only_carbon, 
                "reporting_e_comp": e_comp,
                "reporting_ci": carbon.carbon_intensity_gco2_kwh,
                "reporting_transfer_mb": actual_transfer_mb,
                "reporting_energy_per_gb": energy_per_gb,
                "reporting_sender_ci": sender_ci
            }

        # --- MIN-MAX NORMALIZATION & FINAL SELECTION ---
        if eligible_nodes:
            min_carbon = min(n["raw_carbon"] for n in eligible_nodes.values())
            max_carbon = max(n["raw_carbon"] for n in eligible_nodes.values())

            for node_name, metrics in eligible_nodes.items():

                if self._mode != SchedulingMode.TANS_GREEN:
                    # CarbonEdge-faithful: compute-only carbon, paper Eq. 4
                    sC = 1.0 / (1.0 + metrics["compute_only_carbon"])
                else:
                    # TANS-Green: transfer-aware carbon, min-max normalised
                    if max_carbon == min_carbon:
                        sC = 1.0  # all nodes identical — treat as equally green
                    else:
                        sC = 1.0 - ((metrics["raw_carbon"] - min_carbon) / (max_carbon - min_carbon))

                total = (w.wR * metrics["sR"] + w.wL * metrics["sL"] +
                        w.wP * metrics["sP"] + w.wB * metrics["sB"] + w.wC * sC)

                if total > best_score:
                    best_score = total
                    best_node  = node_name
                    best_comps = {
                        "sR": metrics["sR"], "sL": metrics["sL"],
                        "sP": metrics["sP"], "sB": metrics["sB"], "sC": sC
                    }
                    best_carbon = (
                        metrics["reporting_e_comp"] * metrics["reporting_ci"]
                        + (metrics["reporting_transfer_mb"] / 1024.0)
                        * metrics["reporting_energy_per_gb"]
                        * metrics["reporting_sender_ci"]
                    )
        # Fallback if all nodes are overloaded/ineligible
        if best_node is None:
            best_node = min(self._live, key=lambda n: self._live[n].current_load)
            best_score = 0.0

        return SchedulingDecision(
            selected_node=best_node,
            partition_id=partition.partition_id,
            total_score=round(best_score, 6),
            component_scores={k: round(v, 4) for k, v in best_comps.items()},
            estimated_carbon_gco2=best_carbon,
            mode=self._mode.value,
        )

    
# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    import time as _time

    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s – %(message)s")

    from src.core.model_partitioner import ModelPartitioner, build_default_nodes

    nodes       = build_default_nodes()
    partitioner = ModelPartitioner()
    monitor     = CarbonMonitor(node_profiles=nodes, refresh_interval_s=0)
    scheduler   = TANSScheduler(nodes, monitor, mode=SchedulingMode.TANS_GREEN)
    rep_nodes = [nodes[0], nodes[3], nodes[5]]
    parts = partitioner.partition(rep_nodes, num_partitions=3)

    print("=== TANS-Green Scheduling ===")
    for trial in range(5):
        decisions = scheduler.schedule(parts)
        for d in decisions:
            print(f"  {d}")
        # simulate completions with realistic exec times
        for d, p in zip(decisions, parts):
            exec_ms = p.compute_cost_mflops / 100.0  # rough ms proxy
            scheduler.record_completion(d.selected_node, exec_ms, p.partition_id)
        print()

    print("=== Scheduler snapshot ===")
    print(json.dumps(scheduler.full_snapshot(), indent=2))