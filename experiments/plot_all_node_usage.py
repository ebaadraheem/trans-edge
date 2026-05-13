# plot_all_node_usage.py
"""
Reads node_load_*.csv files from a given directory, groups them by
(model, topology, node_count), and for each group plots the percentage
of partitions assigned to fibre nodes vs. arrival rate for all four
scheduling modes.  Saves one PDF per group.
"""

import pandas as pd
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

COLOURS = {
    "performance": "red",
    "balanced": "orange",
    "green": "green",
    "tans_green": "blue",
}

def compute_fibre_fraction(df, fibre_nodes):
    """
    df: DataFrame with columns: rate, partition_id, node-4, node-6, ...
    Each row gives the % of a single partition assigned to each node.
    Returns a Series indexed by rate with the mean fibre fraction
    across the three partitions (value between 0 and 1).
    """
    # Sum the percentages across fibre nodes for each row,
    # then average over the three partitions per rate.
    # Each row sums to 100% (all nodes), so fibre fraction per row is
    # sum(fibre_nodes) / 100.
    df = df.copy()
    df["fibre_pct"] = df[fibre_nodes].sum(axis=1)          # sum of fibre node percentages
    # Group by rate: average over the three partition rows
    fibre_frac = df.groupby("rate")["fibre_pct"].mean() / 100.0
    return fibre_frac

def plot_group(model, topo, num_nodes, mode_files, output_dir):
    """Plot fibre usage for one (model, topo, num_nodes) group."""
    fibre_nodes = ["node-4", "node-5", "node-6"] if num_nodes == 6 else ["node-4", "node-6"]

    fig, ax = plt.subplots(figsize=(10, 5))
    for mode, filepath in mode_files.items():
        df = pd.read_csv(filepath)
        # Ensure columns are correctly named (strip spaces)
        df.columns = df.columns.str.strip()
        # Drop rows where all node columns are NaN (just in case)
        df = df.dropna(subset=fibre_nodes, how="all")
        frac = compute_fibre_fraction(df, fibre_nodes)
        ax.plot(frac.index, frac.values * 100, label=mode, color=COLOURS[mode])

    ax.set_xlabel("Arrival rate (req/s)")
    ax.set_ylabel("Fibre node usage (%)")
    ax.set_title(f"Fibre node usage – {model} {topo} ({num_nodes} nodes)")
    ax.legend()
    ax.grid(alpha=0.3)
    # Ensure y-axis goes from 0 to 100
    ax.set_ylim(0, 105)

    out_name = output_dir / f"fibre_usage_{model}_{topo}_{num_nodes}nodes.pdf"
    fig.savefig(out_name, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_name}")

def main():
    parser = argparse.ArgumentParser(
        description="Plot fibre node usage from node_load CSV files"
    )
    parser.add_argument(
        "--data_dir", default=".",
        help="Directory containing node_load_*.csv files (default: current directory)"
    )
    parser.add_argument(
        "--output_dir", default="plots",
        help="Directory to save the output PDFs (default: 'plots')"
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all node_load CSV files
    csv_files = sorted(data_dir.glob("node_load_*.csv"))
    if not csv_files:
        print(f"No node_load_*.csv files found in {data_dir.resolve()}")
        return

    # Group by (model, topology, num_nodes)
    groups = {}
    for f in csv_files:
        # Expected filename: node_load_<model>_<topo>_<num_nodes>nodes_<mode>.csv
        stem = f.stem  # e.g., "node_load_resnet50_hetero_6nodes_tans_green"
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
        if len(mode_files) == 4:
            plot_group(model, topo, num_nodes, mode_files, output_dir)
        else:
            print(f"Skipping {model}_{topo}_{num_nodes}nodes – only {len(mode_files)} modes found.")

if __name__ == "__main__":
    main()