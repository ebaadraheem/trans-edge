import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

# Colour and style for each mode
MODE_STYLES = {
    "performance": {"color": "#d62728", "linestyle": "-", "marker": "o"},
    "balanced":    {"color": "#ff7f0e", "linestyle": "--", "marker": "s"},
    "green":       {"color": "#2ca02c", "linestyle": "-.", "marker": "^"},
    "tans_green":  {"color": "#1f77b4", "linestyle": "-", "marker": "D", "linewidth": 2},
}

def plot_sweep(csv_path, output_path=None):
    # Load data
    df = pd.read_csv(csv_path)
    # Convert rate to numeric (in case of errors)
    df["rate"] = pd.to_numeric(df["rate"], errors="coerce")
    for col in ["avg_latency_ms", "p95_latency_ms", "avg_throughput_rps",
                "total_energy_kwh", "total_carbon_gco2", "carbon_efficiency"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.dropna(subset=["rate", "avg_latency_ms", "total_carbon_gco2"], inplace=True)

    rates = sorted(df["rate"].unique())
    modes = df["mode"].unique()

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    ax1, ax2 = axes[0]
    ax3, ax4 = axes[1]

    # 1. Average latency
    for mode in modes:
        subset = df[df["mode"] == mode].sort_values("rate")
        style = MODE_STYLES.get(mode, {})
        ax1.plot(subset["rate"], subset["avg_latency_ms"],
                 label=mode, **style)
    ax1.set_xlabel("Arrival rate (req/s)")
    ax1.set_ylabel("Avg. latency (ms)")
    ax1.set_title("Average Latency vs. Rate")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. P95 latency
    for mode in modes:
        subset = df[df["mode"] == mode].sort_values("rate")
        style = MODE_STYLES.get(mode, {})
        ax2.plot(subset["rate"], subset["p95_latency_ms"],
                 label=mode, **style)
    ax2.set_xlabel("Arrival rate (req/s)")
    ax2.set_ylabel("P95 latency (ms)")
    ax2.set_title("P95 Latency vs. Rate")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Total carbon
    for mode in modes:
        subset = df[df["mode"] == mode].sort_values("rate")
        style = MODE_STYLES.get(mode, {})
        ax3.plot(subset["rate"], subset["total_carbon_gco2"],
                 label=mode, **style)
    ax3.set_xlabel("Arrival rate (req/s)")
    ax3.set_ylabel("Total carbon (g CO₂)")
    ax3.set_title("Total Carbon vs. Rate")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Carbon efficiency
    for mode in modes:
        subset = df[df["mode"] == mode].sort_values("rate")
        style = MODE_STYLES.get(mode, {})
        ax4.plot(subset["rate"], subset["carbon_efficiency"],
                 label=mode, **style)
    ax4.set_xlabel("Arrival rate (req/s)")
    ax4.set_ylabel("Inferences per g CO₂")
    ax4.set_title("Carbon Efficiency vs. Rate")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    fig.tight_layout(pad=2.0)

    # Save or show
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {output_path}")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description="Plot rate‑sweep results")
    parser.add_argument("csv", help="Path to sweep CSV file")
    parser.add_argument("--output", "-o", help="Output image path (.png, .pdf, etc.)")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"Error: {csv_path} not found.")
        return

    plot_sweep(csv_path, args.output)

if __name__ == "__main__":
    main()