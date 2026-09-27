"""Prepares interactive data for the EDA site:
1. site_data/inspector.json (~310 KB): 30 sampled alerts (15 escalated + 15 dismissed)
   stratified across 10 deciles of the main contrast, with transactions [day, amount, type, direction].
2. site_data/gains.json (~2.5 KB): Gains curve of the main contrast rule (101 points, 0% to 100%).
"""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import load_signals, load_transactions
from src.features_v4 import build_features_v4
from src.features_v8 import build_features_v8
from src.txcache import sorted_tx

OUT = Path(__file__).resolve().parent / "site_data"
OUT.mkdir(exist_ok=True)

print("1. Loading raw signals and transactions...")
sig = load_signals("train")
tx = load_transactions("train")

print("2. Computing main contrast (q75)...")
x4 = build_features_v4(tx, sig)
x8 = build_features_v8(x4)
ctr = x8["ctr_q75"]
y = sig.set_index("signal_id")["eskalatsiya"]

# --- Gains Curve ---
print("3. Computing gains curve...")
ctr_filled = ctr.fillna(ctr.median())
df_gains = pd.DataFrame({"ctr": ctr_filled, "y": y}).sort_values("ctr", ascending=False)
total_esc = int(df_gains["y"].sum())
total_alerts = len(df_gains)

gains_pts = []
for p in range(101):
    k = int(round(total_alerts * p / 100))
    esc_caught = int(df_gains["y"].iloc[:k].sum()) if k > 0 else 0
    gains_pts.append({
        "p": p,
        "gains": round(float(esc_caught / total_esc), 4),
        "esc_caught": esc_caught,
        "alerts_checked": k
    })

gains_path = OUT / "gains.json"
with open(gains_path, "w", encoding="utf-8") as f:
    json.dump({"total_alerts": total_alerts, "total_esc": total_esc, "points": gains_pts}, f, indent=1)
print(f"  -> {gains_path.name} ({gains_path.stat().st_size / 1024:.1f} KB)")

# --- Alert Inspector Data ---
print("4. Selecting 30 alerts for inspector...")
m = ctr.notna()
q, bins = pd.qcut(ctr[m], 10, retbins=True, labels=False)
df = pd.DataFrame({"sid": ctr[m].index, "ctr": ctr[m].values, "decile": q + 1, "y": y[m].values})

d = sorted_tx("train")
span = d.groupby("sid")["sb"].agg(["count", "max", "min"])
span["day_span"] = (span["max"] - span["min"]) / 86400
valid_sids = span[(span["count"] >= 150) & (span["day_span"] >= 120)].index
df_valid = df[df["sid"].isin(valid_sids)].copy()

rng = np.random.RandomState(42)
plan_y1 = {1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 2, 10: 2}
plan_y0 = {1: 2, 2: 2, 3: 2, 4: 2, 5: 2, 6: 1, 7: 1, 8: 1, 9: 1, 10: 1}

selected = []
for dec in range(1, 11):
    sub1 = df_valid[(df_valid["decile"] == dec) & (df_valid["y"] == 1)]
    sub0 = df_valid[(df_valid["decile"] == dec) & (df_valid["y"] == 0)]
    selected.append(sub1.sample(n=plan_y1[dec], random_state=rng))
    selected.append(sub0.sample(n=plan_y0[dec], random_state=rng))

res = pd.concat(selected).sort_values(["decile", "y", "ctr"]).reset_index(drop=True)

alerts_dict = []
sids_set = set(res["sid"])
d_sub = d[d["sid"].isin(sids_set)].copy()

for _, row in res.iterrows():
    sid = row["sid"]
    sub_tx = d_sub[d_sub["sid"] == sid].sort_values("sb", ascending=False)
    # tx item: [day_before, amount, type_code (0..3), direction_code (0=in, 1=out)]
    tx_list = [
        [round(float(-r["sb"] / 86400), 3), round(float(r["a"]), 3), int(r["typ"]), int(r["out"])]
        for _, r in sub_tx.iterrows()
    ]
    alerts_dict.append({
        "id": sid,
        "y": int(row["y"]),
        "contrast": round(float(row["ctr"]), 3),
        "decile": int(row["decile"]),
        "tx_count": len(tx_list),
        "tx": tx_list
    })

inspector_data = {
    "decile_bins": [round(float(b), 4) for b in bins],
    "types": ["Card", "Bank transfer", "Cash", "International"],
    "type_colors": ["#5B8DEF", "#DB6C4B", "#22A99A", "#9B7BE0"],
    "alerts": alerts_dict
}

insp_path = OUT / "inspector.json"
with open(insp_path, "w", encoding="utf-8") as f:
    json.dump(inspector_data, f, separators=(",", ":"))
print(f"  -> {insp_path.name} ({insp_path.stat().st_size / 1024:.1f} KB)")
print("Done!")
