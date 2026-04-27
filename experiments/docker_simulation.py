"""
experiments/docker_simulation.py
--------------------------------
Entry point for the CarbonEdge SimPy simulation.

Runs the full pipeline across all four scheduling modes and prints a
side-by-side comparison table, replicating (and extending) the
CarbonEdge paper's Table II experiment.

Usage
-----
    # From the project root:
    PYTHONPATH=. python experiments/docker_simulation.py

    # With a real Electricity Maps API key:
    ELECTRICITY_MAPS_API_KEY=<your-key> PYTHONPATH=. python experiments/docker_simulation.py

    # Fast smoke-test (10 requests):
    PYTHONPATH=. python experiments/docker_simulation.py --quick

CLI flags
---------
--quick       : 10 requests, fast calibration (for CI / smoke tests)
--requests N  : Number of inference requests per mode (default 50)
--rate R      : Arrival rate in req/s (default 0.5)
--seed S      : RNG seed (default 42)
--results DIR : Output directory for CSV files (default ./results)
--mode M      : Run only one mode (performance|balanced|green|tans_green)
--docker      : Run real PyTorch containers
"""

from __future__ import annotations
import argparse
import logging
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

from src.core.execution_engine import CarbonEdgeEngine, SimConfig
from src.core.model_partitioner import NodeProfile, build_default_nodes
from src.core.task_scheduler import SchedulingMode
from src.utils.metrics_logger import RunSummary, compare_runs
from src.utils.docker_manager import DockerNodeManager, node_profile_to_config

log = logging.getLogger("docker_simulation")


# ---------------------------------------------------------------------------
# Node definitions
# ---------------------------------------------------------------------------

def get_nodes() -> list[NodeProfile]:
    """
    Three heterogeneous edge nodes with UK / USA / Sweden regional CI.
    Matches the CarbonEdge paper's High / Medium / Low resource profiles
    mapped to realistic national grid intensities.
    """
    return build_default_nodes()

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CarbonEdge simulation runner")
    p.add_argument("--quick",    action="store_true",
                   help="Smoke-test: 10 requests, fast calibration")
    p.add_argument("--docker",   action="store_true",          
                   help="Run real PyTorch containers")
    p.add_argument("--requests", type=int,   default=50,
                   help="Inference requests per mode (default 50)")
    p.add_argument("--rate",     type=float, default=0.5,
                   help="Arrival rate req/s (default 0.5)")
    p.add_argument("--seed",     type=int,   default=42)
    p.add_argument("--results",  type=Path,  default=Path("results"))
    p.add_argument("--mode",     type=str,   default=None,
                   choices=[m.value for m in SchedulingMode],
                   help="Run only this mode")
    p.add_argument("--verbose",  action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Single-mode runner
# ---------------------------------------------------------------------------

def run_mode(
    mode: SchedulingMode,
    nodes: list[NodeProfile],
    num_requests: int,
    arrival_rate: float,
    seed: int,
    results_dir: Path,
    ms_per_100_mflops: float = 25.0,
    use_docker: bool = False,
) -> RunSummary:
    """Run one complete simulation for *mode*."""
    run_id = f"{mode.value}"
    cfg = SimConfig(
        arrival_rate_rps=arrival_rate,
        num_requests=num_requests,
        mode=mode,
        results_dir=results_dir,
        run_id=run_id,
        seed=seed,
        ms_per_100_mflops=ms_per_100_mflops,
    )

    log.info("=" * 55)
    log.info("  Mode: %-20s  requests=%d", mode.value.upper(), num_requests)
    log.info("=" * 55)

    # --- START DOCKER ---
    docker_mgr = None
    if use_docker:
        configs = [node_profile_to_config(n, base_port=5100 + i) for i, n in enumerate(nodes)]
        docker_mgr = DockerNodeManager(configs, dry_run=False)
        docker_mgr.start_all()
    # -------------------------------

    # Pass the docker_mgr to the engine
    engine = CarbonEdgeEngine(nodes=nodes, config=cfg, docker_manager=docker_mgr)
    
    t0     = time.perf_counter()
    logger, summary = engine.run()
    elapsed = time.perf_counter() - t0
    
    # --- STOP DOCKER ---
    if use_docker:
        docker_mgr.stop_all()
    # ------------------------------
    
    log.info("[main] %s completed in %.2f s (real)", mode.value, elapsed)

    return summary


# ---------------------------------------------------------------------------
# Comparison printer
# ---------------------------------------------------------------------------

def print_comparison(summaries: list[RunSummary], baseline_mode: str = "performance") -> None:
    """
    Print a LaTeX-style ASCII table comparing all modes.
    Highlights carbon reduction vs. the performance (baseline) mode.
    """
    # Find baseline for relative comparisons
    baseline = next((s for s in summaries if s.mode == baseline_mode), summaries[0])

    print("\n" + "╔" + "═" * 94 + "╗")
    print("║  CarbonEdge — Multi-Mode Comparison                                             "
          "              ║")
    print("╠" + "═" * 94 + "╣")
    header = (
        f"  {'Mode':<14} │ {'Avg Lat':>9} │ {'P95 Lat':>9} │ "
        f"{'Tput':>7} │ {'Energy':>12} │ {'Carbon':>12} │ "
        f"{'Eff':>10} │ {'Δ Carbon':>10}"
    )
    print(f"║{header}  ║")
    print("╠" + "═" * 94 + "╣")

    for s in summaries:
        delta_carbon = (
            (s.total_carbon_gco2 - baseline.total_carbon_gco2)
            / baseline.total_carbon_gco2 * 100
            if baseline.total_carbon_gco2 > 0 else 0.0
        )
        sign  = "+" if delta_carbon >= 0 else ""
        delta_str = f"{sign}{delta_carbon:.1f}%"
        flag  = "⬇ " if delta_carbon < -5 else ("⬆ " if delta_carbon > 5 else "  ")

        row = (
            f"  {s.mode:<14} │ "
            f"{s.avg_latency_ms:>8.1f}ms │ "
            f"{s.p95_latency_ms:>8.1f}ms │ "
            f"{s.avg_throughput_rps:>6.2f}r │ "
            f"{s.total_energy_kwh:>11.8f} │ "
            f"{s.total_carbon_gco2:>11.6f} │ "
            f"{s.carbon_efficiency:>9.2f}i │ "
            f"{flag}{delta_str:>9}"
        )
        print(f"║{row}  ║")

    print("╚" + "═" * 94 + "╝")
    print("  Columns: Avg/P95 latency (ms), Throughput (req/s), Energy (kWh),")
    print("  Carbon (gCO₂), Efficiency (inferences/gCO₂), Δ vs Performance mode.")

    # TANS-Green vs CarbonEdge-Green
    tans   = next((s for s in summaries if s.mode == "tans_green"),  None)
    green  = next((s for s in summaries if s.mode == "green"),       None)
    if tans and green and green.total_carbon_gco2 > 0:
        delta = (tans.total_carbon_gco2 - green.total_carbon_gco2) / green.total_carbon_gco2 * 100
        print(f"\n  TANS-Green vs Green: {delta:+.1f}% carbon "
              f"(transfer-awareness impact)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s – %(message)s",
        datefmt="%H:%M:%S",
    )

    nodes       = get_nodes()
    results_dir = args.results
    results_dir.mkdir(parents=True, exist_ok=True)

    num_requests = 10 if args.quick else args.requests
    ms_calib     = 5.0 if args.quick else 25.0  

    if args.mode:
        modes = [SchedulingMode(args.mode)]
    else:
        modes = list(SchedulingMode)

    print(f"\n{'─'*60}")
    print(f"  CarbonEdge Simulation")
    print(f"  Nodes   : {', '.join(n.name for n in nodes)}")
    print(f"  Requests: {num_requests} per mode")
    print(f"  Rate    : {args.rate} req/s")
    print(f"  Modes   : {[m.value for m in modes]}")
    print(f"{'─'*60}\n")

    summaries: list[RunSummary] = []
    for mode in modes:
        summary = run_mode(
            mode=mode,
            nodes=nodes,
            num_requests=num_requests,
            arrival_rate=args.rate,
            seed=args.seed,
            results_dir=results_dir,
            ms_per_100_mflops=ms_calib,
            use_docker=args.docker,
        )
        summaries.append(summary)

    if len(summaries) > 1:
        print_comparison(summaries)

    print(f"\n  CSV detail files → {results_dir.resolve()}/")
    compare_runs(results_dir)


if __name__ == "__main__":
    main()