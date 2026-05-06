
import sys
import csv
import logging
from pathlib import Path

# Make project root importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.execution_engine import CarbonEdgeEngine, SimConfig
from src.core.task_scheduler import SchedulingMode
from src.core.model_partitioner import NodeProfile

# ------------------------------------------------------------
# Configuration – adjust these before running
# ------------------------------------------------------------

# Which model profile to use
PROFILE_CSV = Path("data/resnet50_profile.csv")
MODEL_NAME = "vgg16"          # used for output filename

# Node configuration (uncomment the desired set)
# Heterogeneous nodes (fast CPUs on slow wireless, slower CPUs on fast fiber)
# NODES = [
# NodeProfile(name="node-1", cpu_cores=1.0, ram_gb=2.0, carbon_intensity_gco2_kwh=620, avg_power_w=25, network_type="4g_lte"),
# NodeProfile(name="node-2", cpu_cores=0.9, ram_gb=1.8, carbon_intensity_gco2_kwh=580, avg_power_w=22, network_type="5g"),
# NodeProfile(name="node-3", cpu_cores=0.8, ram_gb=1.6, carbon_intensity_gco2_kwh=450, avg_power_w=18, network_type="wifi"),
# NodeProfile(name="node-4", cpu_cores=0.7, ram_gb=1.4, carbon_intensity_gco2_kwh=350, avg_power_w=15, network_type="fiber"),
# NodeProfile(name="node-5", cpu_cores=0.65, ram_gb=1.2, carbon_intensity_gco2_kwh=250, avg_power_w=15, network_type="fiber"),
# NodeProfile(name="node-6", cpu_cores=0.6, ram_gb=1.0, carbon_intensity_gco2_kwh=200, avg_power_w=15, network_type="fiber"),
 
# ]
NETWORK_TYPE = "homo"

# Homogeneous nodes (all fiber) – uncomment below and comment the above block
NODES = [
NodeProfile(name="node-1", cpu_cores=1.0, ram_gb=2.0, carbon_intensity_gco2_kwh=620, avg_power_w=25, network_type="fiber"),
NodeProfile(name="node-2", cpu_cores=0.9, ram_gb=1.8, carbon_intensity_gco2_kwh=580, avg_power_w=22, network_type="fiber"),
NodeProfile(name="node-3", cpu_cores=0.8, ram_gb=1.6, carbon_intensity_gco2_kwh=450, avg_power_w=18, network_type="fiber"),
NodeProfile(name="node-4", cpu_cores=0.7, ram_gb=1.4, carbon_intensity_gco2_kwh=350, avg_power_w=15, network_type="fiber"),
NodeProfile(name="node-5", cpu_cores=0.65, ram_gb=1.2, carbon_intensity_gco2_kwh=250, avg_power_w=15, network_type="fiber"),
NodeProfile(name="node-6", cpu_cores=0.6, ram_gb=1.0, carbon_intensity_gco2_kwh=200, avg_power_w=15, network_type="fiber"),
]
# NETWORK_TYPE = "homo"

# Simulation settings
NUM_REQUESTS = 100          # requests per run
MS_PER_100_MFLOPS = 6.0     # compute cost calibration
SEED = 42
MAX_CONCURRENT = 1          # crucial for transfer‑trap breakthrough

OUTPUT_CSV = Path(f"sweep_{MODEL_NAME}_{NETWORK_TYPE}.csv")

# ------------------------------------------------------------
# Helper: run one simulation and return summary as dict
# ------------------------------------------------------------
def run_single(mode, nodes, rate):
    run_id = f"{mode.value}_r{rate}"
    cfg = SimConfig(
        arrival_rate_rps=rate,
        num_requests=NUM_REQUESTS,
        mode=mode,
        results_dir=Path("results"),   # detail files will end up here
        run_id=run_id,
        seed=SEED,
        ms_per_100_mflops=MS_PER_100_MFLOPS,
        max_concurrent_per_node=MAX_CONCURRENT,
    )
    engine = CarbonEdgeEngine(nodes=nodes, config=cfg, profile_csv=PROFILE_CSV)
    logger = engine.run()                     # runs simulation and finalizes logger
    summary = logger._build_summary()         # extract RunSummary
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

# ------------------------------------------------------------
# Main sweep
# ------------------------------------------------------------
def main():
    logging.basicConfig(level=logging.WARNING)  # suppress debug output
    modes = list(SchedulingMode)

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "rate", "mode", "avg_latency_ms", "p95_latency_ms",
            "avg_throughput_rps", "total_energy_kwh", "total_carbon_gco2",
            "carbon_efficiency"
        ])
        writer.writeheader()

        for rate in range(1, 101):   # 1 to 100 req/s
            print(f"Rate {rate} req/s")
            for mode in modes:
                try:
                    row = run_single(mode, NODES, rate)
                    writer.writerow(row)
                    f.flush()
                    print(f"  {mode.value:12s} | latency: {row['avg_latency_ms']:7.2f} ms | carbon: {row['total_carbon_gco2']:.4f} g")
                except Exception as e:
                    logging.error(f"Failed rate={rate}, mode={mode.value}: {e}")
                    # write a placeholder row with error indication?
                    writer.writerow({"rate": rate, "mode": mode.value, "avg_latency_ms": "ERROR"})

    print(f"Sweep complete. Results saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()