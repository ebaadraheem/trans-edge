"""
src/core/execution_engine.py
-----------------------------
SimPy discrete-event simulation orchestrator for CarbonEdge.

Architecture
------------
The engine models the following pipeline for each inference request:

  [Arrival] → [TANS Schedule] → [Node Queue] → [Inference] → [Transfer]
      ↓                                                           ↓
  SimPy env                                            MetricsLogger

Key SimPy concepts used
-----------------------
• simpy.Environment   : The discrete-event simulation clock.
• simpy.Resource      : One per edge node, limiting concurrent tasks.
• simpy.Store         : Arrival queue (unbounded).
• simpy.events.Timeout: Models inference latency (scaled from MFLOPs).

Time units: milliseconds (1 SimPy time unit = 1 ms).

Real Docker integration
-----------------------
When docker_manager is provided (not None), the engine calls
docker_manager.run_inference() to execute real containers; otherwise it
uses a SimPy Timeout scaled from compute cost (dry-run / unit-test mode).

The engine supports:
  • Poisson-process arrivals (configurable rate)
  • Dynamic node addition / removal (simulates AMP4EC adaptability)
  • Per-mode carbon accounting via CarbonMonitor
  • Full metrics export via MetricsLogger
"""

from __future__ import annotations

import logging
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import simpy

from src.core.carbon_monitor import CarbonMonitor
from src.core.model_partitioner import ModelPartitioner, NodeProfile, Partition
from src.core.task_scheduler import SchedulingMode, TANSScheduler
from src.utils.metrics_logger import MetricsLogger

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Simulation configuration
# ---------------------------------------------------------------------------

@dataclass
class SimConfig:
    """Top-level simulation parameters."""
    # Arrival process
    arrival_rate_rps: float = 3.0          # average requests per second
    num_requests:     int   = 50           # total inferences to simulate

    # Node resources (SimPy concurrency limits)
    max_concurrent_per_node: int = 2

    # Compute time calibration:
    # 100 MFLOPs on a 1-core node ≈ how many ms of wall-clock time?
    ms_per_100_mflops: float = 25.0

    # Scheduling mode
    mode: SchedulingMode = SchedulingMode.TANS_GREEN

    # Metrics output
    results_dir: Path = Path("results")
    run_id:      Optional[str] = None

    # RNG seed for reproducibility
    seed: int = 42


# ---------------------------------------------------------------------------
# Internal inference request
# ---------------------------------------------------------------------------

@dataclass
class InferenceRequest:
    req_id:       int
    arrival_time: float   # SimPy time (ms)
    partitions:   List[Partition]


# ---------------------------------------------------------------------------
# SimPy Execution Engine
# ---------------------------------------------------------------------------

class CarbonEdgeEngine:
    """
    Discrete-event simulation of CarbonEdge.

    Parameters
    ----------
    nodes          : Edge node profiles.
    config         : Simulation parameters.
    docker_manager : Optional DockerNodeManager for real container calls.
    profile_csv    : Path to resnet50_profile.csv.
    """

    def __init__(
        self,
        nodes: List[NodeProfile],
        config: SimConfig,
        docker_manager=None,
        profile_csv: Path | str = Path(__file__).parents[2] / "data" / "resnet50_profile.csv",
    ) -> None:
        self._nodes   = nodes
        self._cfg     = config
        self._docker  = docker_manager
        self._rng     = random.Random(config.seed)

        # Core components
        self._partitioner = ModelPartitioner(profile_csv)
        self._monitor     = CarbonMonitor(node_profiles=nodes, refresh_interval_s=0)
        self._scheduler   = TANSScheduler(nodes, self._monitor, mode=config.mode)
        self._logger      = MetricsLogger(
            output_dir=config.results_dir,
            run_id=config.run_id,
            mode=config.mode.value,
        )

        rep_nodes = [nodes[0], nodes[3], nodes[5]]
        self._partitions: List[Partition] = self._partitioner.partition(
            rep_nodes, num_partitions=3
        )

        # SimPy environment + per-node resource pools
        self._env:   simpy.Environment = simpy.Environment()
        self._pools: Dict[str, simpy.Resource] = {
            n.name: simpy.Resource(
                self._env,
                capacity=config.max_concurrent_per_node,
            )
            for n in nodes
        }

        # Global stats
        self._completed = 0
        self._failed    = 0

    # ------------------------------------------------------------------
    # Public: run
    # ------------------------------------------------------------------

    def run(self) -> MetricsLogger:
        """
        Execute the full simulation and return the MetricsLogger for
        downstream analysis.
        """
        log.info(
            "[engine] Starting simulation: %d requests @ %.1f rps, mode=%s",
            self._cfg.num_requests, self._cfg.arrival_rate_rps,
            self._cfg.mode.value,
        )

        # Launch the arrival generator process
        self._env.process(self._arrival_generator())

        self._env.run()

        log.info(
            "[engine] Simulation complete: %d completed, %d failed, "
            "sim_time=%.1f ms",
            self._completed, self._failed, self._env.now,
        )

        # Pass the true simulation clock time to the logger for accurate Throughput
        summary = self._logger.finalize(sim_time_ms=self._env.now)
        self._print_summary(summary)
        return self._logger, summary

    # ------------------------------------------------------------------
    # SimPy processes
    # ------------------------------------------------------------------

    def _arrival_generator(self):
        """
        Generates InferenceRequest events following a Poisson process.
        Inter-arrival time ~ Exp(1 / arrival_rate_rps) in milliseconds.
        """
        mean_inter_ms = 1000.0 / self._cfg.arrival_rate_rps

        for req_id in range(self._cfg.num_requests):
            # Schedule an inference pipeline process
            self._env.process(
                self._inference_pipeline(
                    InferenceRequest(
                        req_id=req_id,
                        arrival_time=self._env.now,
                        partitions=self._partitions,
                    )
                )
            )
            # Exponential inter-arrival time
            inter_ms = self._rng.expovariate(1.0 / mean_inter_ms)
            yield self._env.timeout(inter_ms)

    def _inference_pipeline(self, req: InferenceRequest):
        """
        SimPy process: runs one full inference request through all partitions.

        Pipeline:
          For each partition p:
            1. TANS selects a node.
            2. Acquire the node's SimPy Resource (concurrency control).
            3. Execute inference (SimPy Timeout OR real Docker call).
            4. Account for inter-partition transfer latency (aware of co-location).
            5. Record carbon, energy, metrics.
        """
        t_pipeline_start = self._env.now
        total_latency    = 0.0
        total_carbon     = 0.0
        sched_t0         = time.perf_counter()

        # --- TANS scheduling (happens before any resource acquisition) ---
        decisions = self._scheduler.schedule(req.partitions)
        sched_oh_ms = (time.perf_counter() - sched_t0) * 1000

        # We use enumerate here to allow looking ahead to the next decision
        for i, (part, decision) in enumerate(zip(req.partitions, decisions)):
            node_name = decision.selected_node
            pool      = self._pools[node_name]
            node_prof = next(n for n in self._nodes if n.name == node_name)
            ci        = self._monitor.nodes[node_name].carbon_intensity_gco2_kwh

            # ---- resource acquisition (models queuing delay) ----
            with pool.request() as req_token:
                yield req_token       # wait until a slot is free

                # ---- compute ----
                exec_ms = self._exec_time_ms(part, node_prof)
                yield self._env.timeout(exec_ms)

                total_latency += exec_ms

                # ---- record completion in scheduler (updates EMA) ----
                self._scheduler.record_completion(node_name, exec_ms, part.partition_id)

                # ---- compute energy + carbon ----
                e_comp  = self._monitor.nodes[node_name].estimate_inference_energy(exec_ms)
                c_comp  = e_comp * ci

                # ---- inter-partition transfer ----
                # FIX 2: Check if next partition is on the SAME node (Co-location)
                is_colocated = False
                if i + 1 < len(decisions):
                    next_node = decisions[i + 1].selected_node
                    if next_node == node_name:
                        is_colocated = True

                # If co-located, the tensor doesn't move over the network
                xfer_mb    = part.output_tensor_size_mb if not is_colocated else 0.0
                
                # Calculate delays and carbon based on the adjusted transfer size
                xfer_ms    = self._transfer_latency_ms(xfer_mb, node_prof)
                c_trans    = self._monitor.estimate_transfer_carbon(node_name, xfer_mb)
                e_trans_kwh = (xfer_mb / 1024.0) * self._transfer_coeff(node_prof)

                if xfer_ms > 0:
                    yield self._env.timeout(xfer_ms)
                    total_latency += xfer_ms

                total_carbon += c_comp + c_trans

                # ---- log metrics ----
                self._logger.log_inference(
                    request_id=req.req_id,
                    node=node_name,
                    partition_id=part.partition_id,
                    latency_ms=exec_ms + xfer_ms,
                    energy_kwh=e_comp + e_trans_kwh,
                    compute_carbon_gco2=c_comp,
                    transfer_carbon_gco2=c_trans,
                    transfer_size_mb=xfer_mb,
                    network_type=node_prof.network_type,
                    ci_gco2_kwh=ci,
                    scheduling_overhead_ms=sched_oh_ms if part.partition_id == 0
                                           else 0.0,
                )

        self._completed += 1
        log.debug(
            "[engine] req %d done | latency=%.1f ms | carbon=%.6f gCO₂",
            req.req_id, total_latency, total_carbon,
        )

    # ------------------------------------------------------------------
    # Dynamic node events (AMP4EC adaptability feature)
    # ------------------------------------------------------------------

    def add_node(self, node: NodeProfile) -> None:
        """
        Add a new edge node at runtime (simulates a device joining the cluster).
        A new SimPy Resource pool is created; the monitor and scheduler are updated.
        """
        if node.name in self._pools:
            log.warning("[engine] Node %r already present.", node.name)
            return

        self._nodes.append(node)
        self._pools[node.name] = simpy.Resource(
            self._env, capacity=self._cfg.max_concurrent_per_node
        )
        self._monitor.nodes[node.name] = __import__(
            "src.core.carbon_monitor", fromlist=["NodeCarbonState"]
        ).NodeCarbonState(
            node_name=node.name,
            carbon_intensity_gco2_kwh=node.carbon_intensity_gco2_kwh,
            avg_power_w=node.avg_power_w,
            network_type=node.network_type,
        )
        self._scheduler._nodes[node.name] = node
        self._scheduler._live[node.name] = __import__(
            "src.core.task_scheduler", fromlist=["LiveNodeState"]
        ).LiveNodeState(node_name=node.name)

        # Re-partition the model to include the new node
        self._partitions = self._partitioner.partition(
            self._nodes, num_partitions=3
        )
        log.info("[engine] Node %r added; model re-partitioned.", node.name)

    def remove_node(self, node_name: str) -> None:
        """
        Remove a node (simulates device failure / going offline).
        The Resource pool is closed; in-flight requests on that node will
        complete their current timeout before the pool empties.
        """
        if node_name not in self._pools:
            log.warning("[engine] Node %r not found.", node_name)
            return

        self._nodes = [n for n in self._nodes if n.name != node_name]
        del self._pools[node_name]
        del self._scheduler._nodes[node_name]
        del self._scheduler._live[node_name]

        if len(self._nodes) > 0:
            self._partitions = self._partitioner.partition(
                self._nodes, num_partitions=3
            )
        log.info("[engine] Node %r removed; model re-partitioned.", node_name)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _exec_time_ms(self, part: Partition, node: NodeProfile) -> float:
        """
        Estimate execution time for one partition on *node*.

        Base: ms_per_100_mflops × (cost / 100)
        Scaled by node's relative CPU (weaker node → longer).
        Plus small Gaussian jitter for realism.
        """
        max_cpu = max(n.cpu_cores for n in self._nodes) or 1.0
        cpu_factor = max_cpu / node.cpu_cores if node.cpu_cores > 0 else 2.0

        base_ms = self._cfg.ms_per_100_mflops * (part.compute_cost_mflops / 100.0)
        scaled  = base_ms * cpu_factor
        jitter  = self._rng.gauss(0, scaled * 0.05)   # ±5% noise
        return max(1.0, scaled + jitter)

    @staticmethod
    def _transfer_latency_ms(size_mb: float, node: NodeProfile) -> float:
        """
        Model inter-partition transfer latency.
        5G  ≈ 100 Mbps effective → 1 MB ≈ 10 ms (incl. RTT overhead)
        Fiber ≈ 500 Mbps effective → 1 MB ≈  2 ms
        """
        if size_mb <= 0:
            return 0.0
        mbps = 100.0 if node.network_type == "5G" else 500.0
        return size_mb * 8 / mbps * 1000   # ms

    @staticmethod
    def _transfer_coeff(node: NodeProfile) -> float:
        """kWh/GB for this node's link type."""
        from src.core.carbon_monitor import TRANSFER_ENERGY_KWH_PER_GB
        return TRANSFER_ENERGY_KWH_PER_GB.get(
            node.network_type, TRANSFER_ENERGY_KWH_PER_GB["default"]
        )

    @staticmethod
    def _print_summary(summary) -> None:
        print("\n" + "=" * 60)
        print(f"  CarbonEdge Run Summary  —  mode: {summary.mode}")
        print("=" * 60)
        print(f"  Inferences     : {summary.total_inferences}")
        print(f"  Avg latency    : {summary.avg_latency_ms:.2f} ms")
        print(f"  P95 latency    : {summary.p95_latency_ms:.2f} ms")
        print(f"  Avg throughput : {summary.avg_throughput_rps:.3f} req/s")
        print(f"  Total energy   : {summary.total_energy_kwh:.8f} kWh")
        print(f"  Total carbon   : {summary.total_carbon_gco2:.8f} gCO₂")
        print(f"  Carbon effic.  : {summary.carbon_efficiency:.2f} inf/gCO₂")
        print(f"  Sched overhead : {summary.avg_scheduling_oh_ms:.4f} ms")
        print(f"  Duration       : {summary.duration_s:.3f} s (real)")
        print(f"  Nodes used     : {summary.nodes_used}")
        print("=" * 60)


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s – %(message)s",
    )

    from src.core.model_partitioner import build_default_nodes
    import tempfile

    nodes = build_default_nodes()

    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = SimConfig(
            arrival_rate_rps=5.0,
            num_requests=30,
            mode=SchedulingMode.TANS_GREEN,
            results_dir=Path(tmpdir),
            ms_per_100_mflops=5.0,  # fast for test
        )
        engine = CarbonEdgeEngine(nodes=nodes, config=cfg)
        logger = engine.run()
        print("\nCurrent stats:", logger.current_stats())