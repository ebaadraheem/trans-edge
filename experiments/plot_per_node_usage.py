# plot_per_node_usage.py
"""
Reads node_load_*.csv files, groups them by (model, topology, node_count),
and for each group creates a 2x2 figure (one subplot per mode) showing the
average percentage of partitions assigned to each node vs. arrival rate.
Saves one PDF per group.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import argparse
from pathlib import Path

# Fixed colour map for up to 6 nodes – adjust if you have more
NODE_COLOURS = {
    "node-1": "#d62728",   # red
    "node-2": "#ff7f0e",   # orange
    "node-3": "#bcbd22",   # olive
    "node-4": "#2ca02c",   # green
    "node-5": "#17becf",   # cyan
    "node-6": "#9467bd",   # purple
}

MODES = ["performance", "balanced", "green", "tans_green"]

def load_and_aggregate(csv_path):
    """
    Reads a node_load CSV, averages the node percentages across the three
    partitions for each arrival rate. Returns a DataFrame with columns:
    rate, node-1, node-2, ... (whichever nodes appear).
    """
    df = pd.read_csv(csv_path)
    # Ensure column names are stripped
    df.columns = df.columns.str.strip()
    # Identify node columns (everything except 'rate' and 'partition_id')
    node_cols = [c for c in df.columns if c not in ("rate", "partition_id")]
    # Some files may have missing columns (NaN) for unused nodes – fill with 0
    df[node_cols] = df[node_cols].fillna(0)
    # Group by rate, average over the three partitions
    agg = df.groupby("rate")[node_cols].mean()
    return agg

def plot_group(model, topo, num_nodes, mode_files, output_dir):
    """Create a 2x2 figure for one (model, topo, num_nodes) group."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for ax, mode in zip(axes, MODES):
        if mode not in mode_files:
            ax.set_title(f"{mode} – no data")
            continue
        agg = load_and_aggregate(mode_files[mode])
        node_cols = [c for c in agg.columns if c != "rate"]
        for node in node_cols:
            ax.plot(agg.index, agg[node], label=node,
                    color=NODE_COLOURS.get(node, "gray"), linewidth=1.5)
        ax.set_title(mode)
        ax.set_xlabel("Arrival rate (req/s)")
        ax.set_ylabel("Avg. node usage (%)")
        ax.legend(loc="best", fontsize=8)
        ax.grid(alpha=0.3)
        ax.set_ylim(0, 105)

    fig.suptitle(f"Per‑node usage – {model} {topo} ({num_nodes} nodes)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_name = output_dir / f"per_node_usage_{model}_{topo}_{num_nodes}nodes.pdf"
    fig.savefig(out_name, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_name}")

def main():
    parser = argparse.ArgumentParser(
        description="Plot per‑node usage from node_load CSV files"
    )
    parser.add_argument(
        "--data_dir", default=".",
        help="Directory containing node_load_*.csv files (default: current directory)"
    )
    parser.add_argument(
        "--output_dir", default="per_node_plots",
        help="Directory to save the output PDFs (default: 'per_node_plots')"
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(data_dir.glob("node_load_*.csv"))
    if not csv_files:
        print(f"No node_load_*.csv files found in {data_dir.resolve()}")
        return

    # Group files by (model, topo, num_nodes)
    groups = {}
    for f in csv_files:
        stem = f.stem   # e.g., "node_load_resnet50_hetero_6nodes_tans_green"
        parts = stem.split("_")
        if len(parts) < 6:
            print(f"Skipping {f.name}: unexpected name format")
            continue
        model = parts[2]
        topo = parts[3]
        num_nodes_str = parts[4]          # "6nodes" or "3nodes"
        num_nodes = int(num_nodes_str.replace("nodes", ""))
        # The mode may contain underscores (e.g., "tans_green")
        mode = "_".join(parts[5:])
        key = (model, topo, num_nodes)
        groups.setdefault(key, {})[mode] = f

    for (model, topo, num_nodes), mode_files in groups.items():
        if len(mode_files) < 4:
            print(f"Skipping {model}_{topo}_{num_nodes}nodes – only {len(mode_files)} modes found.")
            continue
        plot_group(model, topo, num_nodes, mode_files, output_dir)

if __name__ == "__main__":
    main()