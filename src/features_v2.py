"""Временная динамика внутри окна — основная гипотеза проекта.

Идея: решение специалиста определяется не средним за 180 дней, а ИЗМЕНЕНИЕМ
поведения к концу окна. Всё считается относительно signal_sanasi конкретного
алерта, никаких глобальных статистик -> лика нет.
"""
import numpy as np
import pandas as pd
from .config import ID, DATE

TURLAR = ["karta", "bank_otkazmasi", "naqd", "xalqaro"]
WINDOWS = [7, 30, 90]


def _prep(tx, signals):
    s = signals.set_index(ID)[DATE]
    tx = tx.copy()
    tx["sig_date"] = tx[ID].map(s)
    tx["days_before"] = (tx["sig_date"] - tx["tranzaksiya_vaqti"]).dt.total_seconds() / 86400
    tx["is_out"] = (tx["kirim_chiqim"].astype(str) == "chiqim").astype(np.int8)
    for t in TURLAR:
        tx[f"is_{t}"] = (tx["tranzaksiya_turi"].astype(str) == t).astype(np.int8)
    tx["hour"] = tx["tranzaksiya_vaqti"].dt.hour
    tx["is_night"] = ((tx["hour"] < 6)).astype(np.int8)
    tx["is_weekend"] = (tx["tranzaksiya_vaqti"].dt.dayofweek >= 5).astype(np.int8)
    return tx


def _window_block(tx, w, index):
    """Агрегаты по последним w дням окна + отношение к полному окну."""
    sub = tx[tx["days_before"] <= w]
    g = sub.groupby(ID, observed=True)
    cols = ["is_out", "is_night", "is_weekend"] + [f"is_{t}" for t in TURLAR]
    f = g.agg(
        **{f"w{w}_n": ("miqdor_indeksi", "size"),
           f"w{w}_mean": ("miqdor_indeksi", "mean"),
           f"w{w}_std": ("miqdor_indeksi", "std"),
           f"w{w}_max": ("miqdor_indeksi", "max"),
           f"w{w}_sum": ("miqdor_indeksi", "sum")}
    )
    r = g[cols].mean()
    r.columns = [f"w{w}_r_{c[3:]}" for c in r.columns]
    return f.join(r).reindex(index)


def build_features_v2(tx: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
    tx = _prep(tx, signals)
    idx = signals[ID]

    g = tx.groupby(ID, observed=True)

    # --- полное окно: ночь/выходные/час
    base = g.agg(
        r_night=("is_night", "mean"),
        r_weekend=("is_weekend", "mean"),
        hour_mean=("hour", "mean"),
        hour_std=("hour", "std"),
        recency=("days_before", "min"),          # дней от последней tx до алерта
    ).reindex(idx)

    # --- межтранзакционные интервалы
    tx_sorted = tx.sort_values([ID, "tranzaksiya_vaqti"])
    gap = tx_sorted.groupby(ID, observed=True)["tranzaksiya_vaqti"].diff().dt.total_seconds() / 3600
    tx_sorted["gap_h"] = gap
    gg = tx_sorted.groupby(ID, observed=True)["gap_h"]
    gaps = pd.DataFrame({
        "gap_mean": gg.mean(), "gap_std": gg.std(),
        "gap_max": gg.max(), "gap_median": gg.median(),
    }).reindex(idx)
    gaps["gap_cv"] = gaps["gap_std"] / (gaps["gap_mean"] + 1e-6)

    # --- всплески: максимум транзакций и суммы за скользящие 24ч
    def burst(d):
        t = d["tranzaksiya_vaqti"].values.astype("datetime64[s]").astype(np.int64)
        a = d["miqdor_indeksi"].values
        n = len(t)
        if n < 2:
            return pd.Series({"burst_n": n, "burst_amt": float(a.sum())})
        lo = np.searchsorted(t, t - 86400, side="left")
        cnt = np.arange(n) - lo + 1
        cs = np.concatenate([[0.0], np.cumsum(a)])
        amt = cs[np.arange(n) + 1] - cs[lo]
        return pd.Series({"burst_n": float(cnt.max()), "burst_amt": float(amt.max())})

    bursts = tx_sorted.groupby(ID, observed=True)[["tranzaksiya_vaqti", "miqdor_indeksi"]]\
                      .apply(burst).reindex(idx)

    # --- повторяемость сумм
    rep = g["miqdor_indeksi"].agg(nun="nunique", cnt="size")
    rep["repeat_ratio"] = 1 - rep["nun"] / rep["cnt"]
    rep = rep[["repeat_ratio"]].reindex(idx)

    # --- энтропия по типам транзакций
    ent = g[[f"is_{t}" for t in TURLAR]].mean()
    p = ent.values.clip(1e-9, 1)
    rep["turi_entropy"] = pd.Series(-(p * np.log(p)).sum(axis=1), index=ent.index).reindex(idx).values

    out = base.join([gaps, bursts, rep])

    # --- оконные срезы + отношения recent/baseline
    full_n = g.size().reindex(idx)
    full_mean = g["miqdor_indeksi"].mean().reindex(idx)
    full_naqd = g["is_naqd"].mean().reindex(idx)
    full_out = g["is_out"].mean().reindex(idx)

    for w in WINDOWS:
        blk = _window_block(tx, w, idx)
        # ставка проекта: не абсолютные значения, а сдвиг относительно baseline
        blk[f"w{w}_rate_ratio"] = (blk[f"w{w}_n"] / w) / (full_n / 180 + 1e-6)
        blk[f"w{w}_mean_ratio"] = blk[f"w{w}_mean"] / (full_mean.abs() + 1e-6)
        blk[f"w{w}_naqd_ratio"] = blk[f"w{w}_r_naqd"] / (full_naqd + 1e-6)
        blk[f"w{w}_out_ratio"] = blk[f"w{w}_r_out"] / (full_out + 1e-6)
        out = out.join(blk)

    out.index = idx
    return out.astype(np.float32)
