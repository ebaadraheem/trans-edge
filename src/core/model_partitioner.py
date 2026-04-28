from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import pandas as pd

@dataclass
class NodeProfile:
    
    name: str
    cpu_cores: float
    ram_gb: float
    carbon_intensity_gco2_kwh: float
    avg_power_w: float = 15.0
    network_type: str = "Fiber"


@dataclass
class Partition:
    
    partition_id: int
    node_name: str
    layer_ids: List[int]                 
    layer_names: List[str]
    input_shape: str
    compute_cost_mflops: float
    output_tensor_size_mb: float
    workload_fraction: float = 0.0

    def __repr__(self) -> str:        
        return (
            f"Partition(id={self.partition_id}, node={self.node_name!r}, "
            f"layers={self.layer_ids[0]}–{self.layer_ids[-1]}, "
            f"cost={self.compute_cost_mflops:.1f} MFLOPs, "
            f"xfer={self.output_tensor_size_mb:.3f} MB)"
        )


# ---------------------------------------------------------------------------
# Capability scoring helpers  
# ---------------------------------------------------------------------------

_W_CPU: float = 0.6   # weight for CPU score
_W_MEM: float = 0.4   # weight for memory score


def _node_capability_score(node: NodeProfile,
                            max_cpu: float,
                            max_ram: float) -> float:
    
    c_norm = node.cpu_cores / max_cpu if max_cpu > 0 else 0.0
    m_norm = node.ram_gb    / max_ram if max_ram > 0 else 0.0
    return _W_CPU * c_norm + _W_MEM * m_norm


def _capability_fractions(nodes: List[NodeProfile]) -> List[float]:
    
    max_cpu = max(n.cpu_cores for n in nodes) or 1.0
    max_ram = max(n.ram_gb    for n in nodes) or 1.0

    scores = [_node_capability_score(n, max_cpu, max_ram) for n in nodes]
    total  = sum(scores)

    if total == 0:
        return [1.0 / len(nodes)] * len(nodes)
    return [s / total for s in scores]


# ---------------------------------------------------------------------------
# Load-balance metric
# ---------------------------------------------------------------------------

def _load_balance_metric(partition_costs: List[float]) -> float:
    
    if len(partition_costs) <= 1:
        return 0.0
    avg = sum(partition_costs) / len(partition_costs)
    return sum(abs(c - avg) for c in partition_costs) / len(partition_costs)


# ---------------------------------------------------------------------------
# Core partitioner class
# ---------------------------------------------------------------------------

class ModelPartitioner:

    def __init__(
        self,
        profile_csv: Path | str = Path(__file__).parents[2] / "data" / "resnet50_profile.csv",
        max_refinement_iters: int = 50,
    ) -> None:
        self._df   = self._load_profile(Path(profile_csv))
        self._max_iters = max_refinement_iters

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def partition(
        self,
        nodes: List[NodeProfile],
        num_partitions: Optional[int] = None,
    ) -> List[Partition]:
        
        num_partitions = num_partitions or len(nodes)
        num_partitions = max(1, min(num_partitions, len(self._df)))

        if not nodes:
            raise ValueError("At least one NodeProfile is required.")

        fractions   = _capability_fractions(nodes)
        target_costs = self._compute_targets(fractions, num_partitions)

        # --- greedy boundary placement ---
        boundaries = self._greedy_boundaries(target_costs)

        # --- iterative refinement to reduce L ---
        boundaries = self._refine_boundaries(boundaries)

        # --- build Partition objects ---
        partitions = self._build_partitions(boundaries, nodes, num_partitions)

        self._log_summary(partitions)
        return partitions

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_profile(path: Path) -> pd.DataFrame:
        if not path.exists():
            raise FileNotFoundError(
                f"Profile CSV not found: {path}\n"
                "Run data/generate_profile.py first."
            )
        df = pd.read_csv(path)
        required = {"layer_id", "layer_name", "input_shape", "compute_cost_mflops",
                    "output_size_mb", "cumulative_cost_mflops"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Profile CSV is missing columns: {missing}")
        return df.sort_values("layer_id").reset_index(drop=True)

    def _total_flops(self) -> float:
        return float(self._df["compute_cost_mflops"].sum())

    def _compute_targets(
        self,
        fractions: List[float],
        num_partitions: int,
    ) -> List[float]:
        total = self._total_flops()
        
        if len(fractions) > num_partitions:
            sub_fractions = fractions[:num_partitions]
            norm_factor = sum(sub_fractions)
            if norm_factor > 0:
                fractions = [f / norm_factor for f in sub_fractions]
            else:
                fractions = [1.0 / num_partitions] * num_partitions
                
        n_nodes = len(fractions)
        targets: List[float] = []

        for i in range(num_partitions):
            if i < n_nodes:
                targets.append(fractions[i] * total)
            else:
                remaining_parts = num_partitions - n_nodes
                targets.append(fractions[n_nodes - 1] * total / (remaining_parts + 1))

        return targets

    def _greedy_boundaries(self, targets: List[float]) -> List[int]:
        
        costs  = self._df["compute_cost_mflops"].tolist()
        n_layers = len(costs)
        n_parts  = len(targets)
        boundaries: List[int] = []

        cursor   = 0
        acc_cost = 0.0

        for p_idx, target in enumerate(targets):
            if p_idx == n_parts - 1:
                boundaries.append(n_layers)
                break
            while cursor < n_layers:
                acc_cost += costs[cursor]
                cursor   += 1
                if acc_cost >= target:
                    break
            boundaries.append(cursor)
            acc_cost = 0.0 

        if boundaries[-1] != n_layers:
            boundaries[-1] = n_layers

        return boundaries

    def _refine_boundaries(self, boundaries: List[int]) -> List[int]:
        
        costs        = self._df["compute_cost_mflops"].tolist()
        output_sizes = self._df["output_size_mb"].tolist()  
        n_layers     = len(costs)

        def _evaluate_score(b_list: List[int]) -> float:
            p_costs = self._partition_costs(b_list, costs)
            L = _load_balance_metric(p_costs)
            
            return L

        best_score = _evaluate_score(boundaries)

        for _ in range(self._max_iters):
            improved = False
            for i in range(len(boundaries) - 1):   
                for delta in (-1, +1):
                    new_b = list(boundaries)
                    candidate = new_b[i] + delta

                    lo = (new_b[i - 1] + 1) if i > 0 else 1
                    hi = (new_b[i + 1] - 1) if i < len(new_b) - 2 else n_layers - 1
                    if not (lo <= candidate <= hi):
                        continue

                    new_b[i] = candidate
                    new_score = _evaluate_score(new_b)
                    
                    if new_score < best_score - 1e-6:
                        boundaries = new_b
                        best_score = new_score
                        improved   = True
                        break
                if improved:
                    break
            if not improved:
                break

        return boundaries

    @staticmethod
    def _partition_costs(
        boundaries: List[int],
        costs: List[float],
    ) -> List[float]:
        starts = [0] + boundaries[:-1]
        return [
            sum(costs[s:e])
            for s, e in zip(starts, boundaries)
        ]

    def _build_partitions(
        self,
        boundaries: List[int],
        nodes: List[NodeProfile],
        num_partitions: int,
    ) -> List[Partition]:
        total_flops = self._total_flops()
        costs       = self._df["compute_cost_mflops"].tolist()
        partitions: List[Partition] = []

        starts = [0] + boundaries[:-1]
        for p_idx, (start, end) in enumerate(zip(starts, boundaries)):
            node     = nodes[min(p_idx, len(nodes) - 1)]
            layer_slice = self._df.iloc[start:end]

            p_cost = float(layer_slice["compute_cost_mflops"].sum())

            if not layer_slice.empty:
                xfer_mb = float(layer_slice.iloc[-1]["output_size_mb"])
            else:
                xfer_mb = 0.0

            if p_idx == num_partitions - 1:
                xfer_mb = 0.0

            partitions.append(Partition(
                partition_id          = p_idx,
                node_name             = node.name,
                layer_ids             = layer_slice["layer_id"].tolist(),
                layer_names           = layer_slice["layer_name"].tolist(),
                compute_cost_mflops   = round(p_cost, 4),
                input_shape           = str(layer_slice.iloc[0]["input_shape"]),
                output_tensor_size_mb = round(xfer_mb, 6),
                workload_fraction     = round(p_cost / total_flops, 4)
                                        if total_flops > 0 else 0.0,
            ))

        return partitions

    @staticmethod
    def _log_summary(partitions: List[Partition]) -> None:  
        total = sum(p.compute_cost_mflops for p in partitions)
        costs = [p.compute_cost_mflops for p in partitions]
        L     = _load_balance_metric(costs)
        print(f"[partitioner] {len(partitions)} partitions | "
              f"total={total:.1f} MFLOPs | L={L:.2f}")
        for p in partitions:
            print(f"  {p}")


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def build_default_nodes() -> List[NodeProfile]:
    return [
        # # Baseline Validation (Control) [1.0, 2.5, 3.5 req/s]
        # NodeProfile(name="node-1",  cpu_cores=0.4, ram_gb=0.5,
        #             carbon_intensity_gco2_kwh=380.0, avg_power_w=8.0,  network_type="fiber"),
        # NodeProfile(name="node-2",  cpu_cores=0.4, ram_gb=0.5,
        #             carbon_intensity_gco2_kwh=420.0, avg_power_w=9.0,  network_type="fiber"),
        # NodeProfile(name="node-3",  cpu_cores=0.6, ram_gb=1.0,
        #             carbon_intensity_gco2_kwh=490.0, avg_power_w=12.0, network_type="fiber"),
        # NodeProfile(name="node-4",  cpu_cores=0.6, ram_gb=1.0,
        #             carbon_intensity_gco2_kwh=530.0, avg_power_w=14.0, network_type="fiber"),
        # NodeProfile(name="node-5", cpu_cores=1.0, ram_gb=2.0,
        #             carbon_intensity_gco2_kwh=580.0, avg_power_w=20.0, network_type="fiber"),
        # NodeProfile(name="node-6", cpu_cores=1.0, ram_gb=2.0,
        #             carbon_intensity_gco2_kwh=620.0, avg_power_w=22.0, network_type="fiber"),
        
        
        
        # # TANS Showcase (Case Study) [1.0, 2.5, 3.5 req/s]
        # NodeProfile(
        #     name="node-1", 
        #     cpu_cores=0.3,
        #     ram_gb=0.512,
        #     carbon_intensity_gco2_kwh=650.0,
        #     avg_power_w=5.0,
        #     network_type="4g_lte",
        # ),
        # NodeProfile(
        #     name="node-2", 
        #     cpu_cores=0.4, 
        #     ram_gb=0.512,
        #     carbon_intensity_gco2_kwh=600.0,
        #     avg_power_w=8.0,
        #     network_type="5g",
        # ),
        # NodeProfile(
        #     name="node-3", 
        #     cpu_cores=0.4, 
        #     ram_gb=0.512,
        #     carbon_intensity_gco2_kwh=550.0,
        #     avg_power_w=8.0,
        #     network_type="wifi",
        # ),
        # NodeProfile(
        #     name="node-4", 
        #     cpu_cores=0.6, 
        #     ram_gb=1.0,
        #     carbon_intensity_gco2_kwh=400.0,
        #     avg_power_w=15.0,
        #     network_type="fiber",
        # ),
        # NodeProfile(
        #     name="node-5",
        #     cpu_cores=1.0, 
        #     ram_gb=2.0,
        #     carbon_intensity_gco2_kwh=380.0,
        #     avg_power_w=25.0,
        #     network_type="fiber",
        # ),
        # NodeProfile(
        #     name="node-6", 
        #     cpu_cores=1.0, 
        #     ram_gb=2.0,
        #     carbon_intensity_gco2_kwh=15.0,
        #     avg_power_w=25.0,
        #     network_type="fiber",
        # ),
        
        
        
        # Overload Stress Test [3.5 req/s]
        NodeProfile(name="node-1", cpu_cores=0.4, ram_gb=0.5,
                    carbon_intensity_gco2_kwh=580.0, avg_power_w=5.0,  network_type="4g_lte"),
        NodeProfile(name="node-2", cpu_cores=0.4, ram_gb=0.5,
                    carbon_intensity_gco2_kwh=540.0, avg_power_w=6.0,  network_type="5g"),
        NodeProfile(name="node-3", cpu_cores=0.6, ram_gb=1.0,
                    carbon_intensity_gco2_kwh=450.0, avg_power_w=10.0, network_type="wifi"),
        NodeProfile(name="node-4", cpu_cores=0.6, ram_gb=1.0,
                    carbon_intensity_gco2_kwh=380.0, avg_power_w=12.0, network_type="fiber"),
        NodeProfile(name="node-5", cpu_cores=0.8, ram_gb=1.5,
                    carbon_intensity_gco2_kwh=280.0, avg_power_w=15.0, network_type="fiber"),
        NodeProfile(name="node-6", cpu_cores=0.8, ram_gb=1.5,
                    carbon_intensity_gco2_kwh=200.0, avg_power_w=18.0, network_type="fiber"),
    ]


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    nodes      = build_default_nodes()
    partitioner = ModelPartitioner()
    rep_nodes = [nodes[0], nodes[3], nodes[5]]
    parts       = partitioner.partition(rep_nodes, num_partitions=3)

    print("\nDetailed partition report:")
    for p in parts:
        print(f"  Partition {p.partition_id} → {p.node_name}")
        print(f"    Layers : {p.layer_names[0]} … {p.layer_names[-1]}")
        print(f"    FLOPs  : {p.compute_cost_mflops:.1f} MFLOPs "
              f"({p.workload_fraction*100:.1f}%)")
        print(f"    Xfer   : {p.output_tensor_size_mb:.4f} MB to next node")