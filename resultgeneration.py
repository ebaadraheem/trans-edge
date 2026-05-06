import pandas as pd

for mode in ["performance","balanced","green","tans_green"]:
    df = pd.read_csv(f"results/{mode}_detail.csv")
    # each inference_id appears 3 times (once per partition)
    inf_lat = df.groupby("inference_id")["latency_ms"].sum().mean()
    total_carbon = df["carbon_gco2"].sum()
    print(f"{mode:>12}: inf_lat={inf_lat:.1f} ms, total_carbon={total_carbon:.6f} g")