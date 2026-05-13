# analyze_node_usage.py
import pandas as pd
from pathlib import Path

def analyze(model, topo, num_nodes, mode, rates=range(1, 101)):
    results_dir = Path("results")
    all_rows = []

    for rate in rates:
        # run_id format from rate_sweep.py: f"{mode.value}_r{rate}"
        file_name = f"{mode}_r{rate}_node_usage.csv"
        file_path = results_dir / file_name
        if not file_path.exists():
            print(f"  [skip] {file_name} not found")
            continue

        df = pd.read_csv(file_path)
        # Count assignments per node per partition for this rate
        counts = df.groupby(["partition_id", "node"]).size().unstack(fill_value=0)
        pct = counts.div(counts.sum(axis=1), axis=0) * 100

        # Build a row: rate, partition_id, then one column per node
        for part_id, row in pct.iterrows():
            record = {"rate": rate, "partition_id": part_id}
            for node_name, val in row.items():
                record[node_name] = round(val, 2)
            all_rows.append(record)

    if not all_rows:
        print("No node-usage files found.")
        return

    out_df = pd.DataFrame(all_rows)
    out_df = out_df.sort_values(["rate", "partition_id"])
    out_path = f"node_load_{model}_{topo}_{num_nodes}nodes_{mode}.csv"
    out_df.to_csv(out_path, index=False)
    print(f"Saved → {out_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="resnet50")
    parser.add_argument("--topo", default="hetero")
    parser.add_argument("--num_nodes", type=int, default=6)
    parser.add_argument("--mode", default="performance")
    parser.add_argument("--all_modes", action="store_true",
                        help="If set, run for all four modes: performance, balanced, green, tans_green")
    args = parser.parse_args()

    if args.all_modes:
        modes = ["performance", "balanced", "green", "tans_green"]
    else:
        modes = [args.mode]

    for m in modes:
        analyze(args.model, args.topo, args.num_nodes, m)