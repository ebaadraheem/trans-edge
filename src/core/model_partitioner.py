from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import pandas as pd

@dataclass
class NodeProfile:
    """
    Describes one edge node's capabilities and regional carbon context.

    Attributes
    ----------
    name          : Unique node identifier, e.g. "node-uk".
    cpu_cores     : Fractional CPU allocation (e.g. 1.0 = one full core).
    ram_gb        : RAM limit in gigabytes.
    carbon_intensity_gco2_kwh : Regional grid carbon intensity (gCO₂/kWh).
    avg_power_w   : Node's average power draw in watts (used by TANS).
    network_type  : "5G" | "Fiber" – determines transfer energy coefficient.
    """
    name: str
    cpu_cores: float
    ram_gb: float
    carbon_intensity_gco2_kwh: float
    avg_power_w: float = 15.0
    network_type: str = "Fiber"


@dataclass
class Partition:
    """
    One logical slice of the ResNet-50 computation graph assigned to a node.
    """
    partition_id: int
    node_name: str
    layer_ids: List[int]                  # indices into the profile CSV
    layer_names: List[str]
    input_shape: str
    compute_cost_mflops: float
    # Serialised tensor that must travel to the *next* node (0 for last)
    output_tensor_size_mb: float
    # Fraction of total model FLOPs (informational)
    workload_fraction: float = 0.0

    def __repr__(self) -> str:           # pragma: no cover
        return (
            f"Partition(id={self.partition_id}, node={self.node_name!r}, "
            f"layers={self.layer_ids[0]}–{self.layer_ids[-1]}, "
            f"cost={self.compute_cost_mflops:.1f} MFLOPs, "
            f"xfer={self.output_tensor_size_mb:.3f} MB)"
        )


# ---------------------------------------------------------------------------
# Capability scoring helpers  (Equations 3–4 from AMP4EC)
# ---------------------------------------------------------------------------

_W_CPU: float = 0.6   # weight for CPU score
_W_MEM: float = 0.4   # weight for memory score


def _node_capability_score(node: NodeProfile,
                            max_cpu: float,
                            max_ram: float) -> float:
    """
    Eq. 3:  Sᵢ = w_cpu · (cᵢ / c_max) + w_mem · (mᵢ / m_max)

    Normalising by the cluster maximum ensures scores ∈ [0, 1].
    """
    c_norm = node.cpu_cores / max_cpu if max_cpu > 0 else 0.0
    m_norm = node.ram_gb    / max_ram if max_ram > 0 else 0.0
    return _W_CPU * c_norm + _W_MEM * m_norm


def _capability_fractions(nodes: List[NodeProfile]) -> List[float]:
    """
    Eq. 4:  Pᵢ = Sᵢ / Σⱼ Sⱼ

    Returns normalised fractions that sum to 1.0, one per node.
    If all scores are zero, falls back to uniform distribution.
    """
    max_cpu = max(n.cpu_cores for n in nodes) or 1.0
    max_ram = max(n.ram_gb    for n in nodes) or 1.0

    scores = [_node_capability_score(n, max_cpu, max_ram) for n in nodes]
    total  = sum(scores)

    if total == 0:
        return [1.0 / len(nodes)] * len(nodes)
    return [s / total for s in scores]


# ---------------------------------------------------------------------------
# Load-balance metric  (Equation 5 from AMP4EC)
# ---------------------------------------------------------------------------

def _load_balance_metric(partition_costs: List[float]) -> float:
    """
    Eq. 5:  L = (1/n) Σᵢ |Lᵢ − L_avg|

    Lower is better.  Returns 0 for a single partition.
    """
    if len(partition_costs) <= 1:
        return 0.0
    avg = sum(partition_costs) / len(partition_costs)
    return sum(abs(c - avg) for c in partition_costs) / len(partition_costs)


# ---------------------------------------------------------------------------
# Core partitioner class
# ---------------------------------------------------------------------------

class ModelPartitioner:
    """
    Greedy RALOS partitioner for ResNet-50.

    Parameters
    ----------
    profile_csv : Path to data/resnet50_profile.csv.
    max_refinement_iters : Upper bound on boundary-nudge iterations.
    """

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
        """
        Split ResNet-50 layers across *nodes* using RALOS.

        Parameters
        ----------
        nodes           : Ordered list of NodeProfile objects (edge cluster).
        num_partitions  : Desired number of cuts. Defaults to len(nodes).

        Returns
        -------
        List[Partition] ordered from first to last sub-model.
        Len == num_partitions (or fewer if the model has too few layers).
        """
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
        """
        Map node capability fractions onto per-partition FLOPs targets.
        If num_partitions < len(nodes), normalize the top fractions to sum to 1.0.
        """
        total = self._total_flops()
        
        # FIX 1: Normalize fractions if there are more nodes than partitions
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
            node_idx = min(i, n_nodes - 1)
            # If more partitions than nodes, split the last node's budget
            if i < n_nodes:
                targets.append(fractions[i] * total)
            else:
                # redistribute remaining budget equally among overflow parts
                remaining_parts = num_partitions - n_nodes
                targets.append(fractions[n_nodes - 1] * total / (remaining_parts + 1))

        return targets

    def _greedy_boundaries(self, targets: List[float]) -> List[int]:
        """
        Place boundary layer indices so each partition accumulates ≈ target.

        Returns a list of *exclusive end* indices, length == num_partitions.
        Last entry is always len(df).
        """
        costs  = self._df["compute_cost_mflops"].tolist()
        n_layers = len(costs)
        n_parts  = len(targets)
        boundaries: List[int] = []

        cursor   = 0
        acc_cost = 0.0

        for p_idx, target in enumerate(targets):
            if p_idx == n_parts - 1:
                # absorb all remaining layers into the last partition
                boundaries.append(n_layers)
                break
            while cursor < n_layers:
                acc_cost += costs[cursor]
                cursor   += 1
                if acc_cost >= target:
                    break
            boundaries.append(cursor)
            acc_cost = 0.0  # reset for next partition

        # safety: ensure we always end at n_layers
        if boundaries[-1] != n_layers:
            boundaries[-1] = n_layers

        return boundaries

    def _refine_boundaries(self, boundaries: List[int]) -> List[int]:
        """
        Nudge boundaries left / right to minimise L (Eq. 5).
        Early-stops when no improvement is found.
        """
        costs    = self._df["compute_cost_mflops"].tolist()
        n_layers = len(costs)
        best_L   = _load_balance_metric(
            self._partition_costs(boundaries, costs)
        )

        for _ in range(self._max_iters):
            improved = False
            for i in range(len(boundaries) - 1):     # never move the last boundary
                for delta in (-1, +1):
                    new_b = list(boundaries)
                    candidate = new_b[i] + delta

                    # keep strictly increasing and within range
                    lo = (new_b[i - 1] + 1) if i > 0 else 1
                    hi = (new_b[i + 1] - 1) if i < len(new_b) - 2 else n_layers - 1
                    if not (lo <= candidate <= hi):
                        continue

                    new_b[i] = candidate
                    new_L = _load_balance_metric(
                        self._partition_costs(new_b, costs)
                    )
                    if new_L < best_L - 1e-6:
                        boundaries = new_b
                        best_L     = new_L
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
        """Sum of compute costs within each boundary slice."""
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

            # Transfer tensor = last layer's output in this partition
            if not layer_slice.empty:
                xfer_mb = float(layer_slice.iloc[-1]["output_size_mb"])
            else:
                xfer_mb = 0.0

            # Last partition has no downstream transfer
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
    def _log_summary(partitions: List[Partition]) -> None:  # pragma: no cover
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
    """
     six diverse nodes with following characteristics:
    - node-1: Deep Edge (Highway) — low capability, high carbon, 4G
    - node-2: Urban Edge (Lahore 5G) — medium capability, medium carbon, 5G
    - node-3: Smart City Edge (Gujranwala Wi-Fi) — medium capability, medium carbon, Wi-Fi
    - node-4: Regional Datacenter (Islamabad) — high capability, medium carbon, Fiber
    - node-5: Standard Cloud (AWS) — very high capability, lower carbon, Fiber
    - node-6: Green Cloud (Sweden) — very high capability, very low carbon, Fiber
    """
    return [
        NodeProfile(
            name="node-1", # Deep Edge (Highway)
            cpu_cores=0.3, # 30000 quota
            ram_gb=0.512,
            carbon_intensity_gco2_kwh=650.0,
            avg_power_w=5.0,
            network_type="4g_lte",
        ),
        NodeProfile(
            name="node-2", # Urban Edge (Lahore 5G)
            cpu_cores=0.4, # 40000 quota
            ram_gb=0.512,
            carbon_intensity_gco2_kwh=600.0,
            avg_power_w=8.0,
            network_type="5g",
        ),
        NodeProfile(
            name="node-3", # Smart City Edge (Gujranwala Wi-Fi)
            cpu_cores=0.4, # 40000 quota
            ram_gb=0.512,
            carbon_intensity_gco2_kwh=550.0,
            avg_power_w=8.0,
            network_type="wifi",
        ),
        NodeProfile(
            name="node-4", # Regional Datacenter (Islamabad)
            cpu_cores=0.6, # 60000 quota
            ram_gb=1.0,
            carbon_intensity_gco2_kwh=400.0,
            avg_power_w=15.0,
            network_type="fiber",
        ),
        NodeProfile(
            name="node-5", # Standard Cloud (AWS)
            cpu_cores=1.0, # 100000 quota
            ram_gb=2.0,
            carbon_intensity_gco2_kwh=380.0,
            avg_power_w=25.0,
            network_type="fiber",
        ),
        NodeProfile(
            name="node-6", # Green Cloud (Sweden)
            cpu_cores=1.0, # 100000 quota
            ram_gb=2.0,
            carbon_intensity_gco2_kwh=15.0,
            avg_power_w=25.0,
            network_type="fiber",
        ),
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