import sys, csv, logging, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.execution_engine import CarbonEdgeEngine, SimConfig
from src.core.task_scheduler import SchedulingMode
from src.core.model_partitioner import NodeProfile


# NodeProfile(name="node-1", cpu_cores=1.0, ram_gb=2.0, carbon_intensity_gco2_kwh=620, avg_power_w=25, network_type="4g_lte")

# --- Helper to define nodes ---
def get_nodes(network_type, num_nodes=6):
    if num_nodes == 6:
        if network_type == "hetero":
            return [
                NodeProfile("node-1", 1.0, 2.0, carbon_intensity_gco2_kwh=620, avg_power_w=25, network_type="4g_lte"),
                NodeProfile("node-2", 0.9, 1.8, carbon_intensity_gco2_kwh=580, avg_power_w=22, network_type="5g"),
                NodeProfile("node-3", 0.8, 1.6, carbon_intensity_gco2_kwh=450, avg_power_w=18, network_type="wifi"),
                NodeProfile("node-4", 0.7, 1.4, carbon_intensity_gco2_kwh=350, avg_power_w=15, network_type="fiber"),
                NodeProfile("node-5", 0.65, 1.2, carbon_intensity_gco2_kwh=250, avg_power_w=15, network_type="fiber"),
                NodeProfile("node-6", 0.6, 1.0, carbon_intensity_gco2_kwh=200, avg_power_w=15, network_type="fiber"),
            ]
        else:  # homo
            return [
                NodeProfile("node-1", 1.0, 2.0, carbon_intensity_gco2_kwh=620, avg_power_w=25, network_type="fiber"),
                NodeProfile("node-2", 0.9, 1.8, carbon_intensity_gco2_kwh=580, avg_power_w=22, network_type="fiber"),
                NodeProfile("node-3", 0.8, 1.6, carbon_intensity_gco2_kwh=450, avg_power_w=18, network_type="fiber"),
                NodeProfile("node-4", 0.7, 1.4, carbon_intensity_gco2_kwh=350, avg_power_w=15, network_type="fiber"),
                NodeProfile("node-5", 0.65, 1.2, carbon_intensity_gco2_kwh=250, avg_power_w=15, network_type="fiber"),
                NodeProfile("node-6", 0.6, 1.0, carbon_intensity_gco2_kwh=200, avg_power_w=15, network_type="fiber"),
            ]
    elif num_nodes == 3:
        # For 3-node experiments, pick nodes 1, 4, 6 from the hetero or homo sets
        if network_type == "hetero":
            return [
                NodeProfile("node-1", 1.0, 2.0, carbon_intensity_gco2_kwh=620, avg_power_w=25, network_type="4g_lte"),
                NodeProfile("node-4", 0.7, 1.4, carbon_intensity_gco2_kwh=350, avg_power_w=15, network_type="fiber"),
                NodeProfile("node-6", 0.6, 1.0, carbon_intensity_gco2_kwh=200, avg_power_w=15, network_type="fiber"),
            ]
        else:  # homo 3 nodes all fiber
            return [
                NodeProfile("node-1", 1.0, 2.0, carbon_intensity_gco2_kwh=620, avg_power_w=25, network_type="fiber"),
                NodeProfile("node-4", 0.7, 1.4, carbon_intensity_gco2_kwh=350, avg_power_w=15, network_type="fiber"),
                NodeProfile("node-6", 0.6, 1.0, carbon_intensity_gco2_kwh=200, avg_power_w=15, network_type="fiber"),
            ]
    else:
        raise ValueError("num_nodes must be 3 or 6")

# --- Run a single simulation ---
def run_single(mode, nodes, rate, profile_csv, num_requests=500, ms_per_100_mflops=3.0, max_conc=2, seed=42):
    run_id = f"{mode.value}_r{rate}"
    cfg = SimConfig(
        arrival_rate_rps=rate,
        num_requests=num_requests,
        mode=mode,
        results_dir=Path("results"),
        run_id=run_id,
        seed=seed,
        ms_per_100_mflops=ms_per_100_mflops,
        max_concurrent_per_node=max_conc,
    )
    engine = CarbonEdgeEngine(nodes=nodes, config=cfg, profile_csv=profile_csv)
    logger = engine.run()
    summary = logger._build_summary()
    return {
        "rate": rate,
        "mode": mode.value,
        "avg_latency_ms": round(summary.avg_latency_ms, 2),
        "p95_latency_ms": round(summary.p95_latency_ms, 2),
        "avg_throughput_rps": round(summary.avg_throughput_rps, 2),
        "total_energy_kwh": round(summary.total_energy_kwh, 10),
        "total_carbon_gco2": round(summary.total_carbon_gco2, 6),
        "carbon_efficiency": round(summary.carbon_efficiency, 2),
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True, help="Path to profile CSV")
    parser.add_argument("--model", required=True, help="Model name for output file")
    parser.add_argument("--topo", required=True, choices=["hetero", "homo"])
    parser.add_argument("--num_nodes", type=int, default=6, choices=[3,6])
    parser.add_argument("--requests", type=int, default=500)
    parser.add_argument("--ms_per_100", type=float, default=3.0)
    parser.add_argument("--max_conc", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)
    nodes = get_nodes(args.topo, args.num_nodes)
    profile_csv = Path(args.profile)
    modes = list(SchedulingMode)
    out_file = Path(f"sweep_{args.model}_{args.topo}_{args.num_nodes}nodes.csv")

    with open(out_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "rate","mode","avg_latency_ms","p95_latency_ms",
            "avg_throughput_rps","total_energy_kwh","total_carbon_gco2","carbon_efficiency"
        ])
        writer.writeheader()
        for rate in range(1, 101):
            print(f"Rate {rate} req/s")
            for mode in modes:
                try:
                    row = run_single(mode, nodes, rate, profile_csv,
                                     num_requests=args.requests,
                                     ms_per_100_mflops=args.ms_per_100,
                                     max_conc=args.max_conc,
                                     seed=args.seed)
                    writer.writerow(row)
                    f.flush()
                    print(f"  {mode.value:12s} | lat={row['avg_latency_ms']:7.2f} ms | carbon={row['total_carbon_gco2']:.4f} g")
                except Exception as e:
                    logging.error(f"Failed rate={rate}, mode={mode.value}: {e}")
                    writer.writerow({"rate": rate, "mode": mode.value, "avg_latency_ms": "ERROR"})
    print(f"Sweep complete → {out_file}")

if __name__ == "__main__":
    main()