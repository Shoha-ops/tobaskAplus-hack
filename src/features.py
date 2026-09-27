"""Transaction aggregation (transaction level) -> features (alert level).

Baseline feature set. All features are computed STRICTLY within each signal_id;
no global dataset-wide target statistics -> zero train/test leakage.
"""
import numpy as np
import pandas as pd
from .config import ID, DATE


TURLAR = ["karta", "bank_otkazmasi", "naqd", "xalqaro"]


def _amount_stats(g, prefix):
    """Statistics of miqdor_indeksi across grouped transactions."""
    out = g["miqdor_indeksi"].agg(
        ["count", "mean", "std", "min", "max", "median", "sum", "skew"]
    )
    out.columns = [f"{prefix}_{c}" for c in out.columns]
    q = g["miqdor_indeksi"].quantile([0.10, 0.25, 0.75, 0.90, 0.99]).unstack()
    q.columns = [f"{prefix}_q{int(c * 100):02d}" for c in q.columns]
    return out.join(q)


def build_features(tx: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
    tx = tx.copy()
    tx["is_out"] = (tx["kirim_chiqim"].astype(str) == "chiqim").astype(np.int8)
    for t in TURLAR:
        tx[f"is_{t}"] = (tx["tranzaksiya_turi"].astype(str) == t).astype(np.int8)
    tx["day"] = tx["tranzaksiya_vaqti"].values.astype("datetime64[D]")

    g = tx.groupby(ID, observed=True)

    # --- overall amount statistics
    feats = _amount_stats(g, "amt")

    # --- direction and type ratios
    ratios = g[["is_out"] + [f"is_{t}" for t in TURLAR]].mean()
    ratios.columns = [f"r_{c[3:]}" for c in ratios.columns]
    feats = feats.join(ratios)

    # raw type counts (for xalqaro, raw count performed better than ratio)
    counts = g[[f"is_{t}" for t in TURLAR]].sum()
    counts.columns = [f"n_{c[3:]}" for c in counts.columns]
    feats = feats.join(counts)

    # --- separate incoming and outgoing statistics
    for flag, prefix in [(0, "in"), (1, "out")]:
        sub = tx[tx["is_out"] == flag]
        s = _amount_stats(sub.groupby(ID, observed=True), prefix)
        feats = feats.join(s)

    feats["bal_sum"] = feats["in_sum"].fillna(0) - feats["out_sum"].fillna(0)
    feats["bal_ratio"] = feats["in_sum"].fillna(0) / (feats["out_sum"].abs().fillna(0) + 1e-6)
    feats["out_in_cnt_ratio"] = feats["out_count"].fillna(0) / (feats["in_count"].fillna(0) + 1e-6)

    # --- temporal activity features
    act = g.agg(
        n_days=("day", "nunique"),
        t_min=("tranzaksiya_vaqti", "min"),
        t_max=("tranzaksiya_vaqti", "max"),
    )
    act["span_days"] = (act["t_max"] - act["t_min"]).dt.total_seconds() / 86400
    act["tx_per_day"] = feats["amt_count"] / act["n_days"].clip(lower=1)
    act["density"] = act["n_days"] / act["span_days"].clip(lower=1)
    feats = feats.join(act[["n_days", "span_days", "tx_per_day", "density"]])

    # --- alert-level features
    s = signals.set_index(ID)
    feats = feats.join(s[[DATE]])
    feats["sig_month"] = feats[DATE].dt.month
    feats["sig_dow"] = feats[DATE].dt.dayofweek
    feats["sig_doy"] = feats[DATE].dt.dayofyear
    feats["sig_days_since_start"] = (feats[DATE] - pd.Timestamp("2025-01-01")).dt.days
    feats = feats.drop(columns=[DATE])

    # row order matches signals order
    feats = feats.reindex(s.index)
    return feats.astype(np.float32)
