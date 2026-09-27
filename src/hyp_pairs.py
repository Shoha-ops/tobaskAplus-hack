"""П1: связи приход -> расход (и расход -> приход).
Для каждой операции берётся ПОСЛЕДНЯЯ операция противоположного направления
ДО неё у того же клиента. Только прошлое внутри окна алерта -> будущего нет.
miqdor_indeksi — общая лог-шкала, поэтому разность сумм = лог отношения сумм."""
import numpy as np, pandas as pd
from .txcache import TYPES

def _prev_opposite(d, target_out):
    """Для строк с out==target_out: индекс последней строки противоположного
    направления раньше в том же алерте."""
    n = len(d); idx = np.arange(n, dtype=float)
    src = np.where(d["out"].values == (1 - target_out), idx, np.nan)
    prev = pd.Series(src).groupby(d["sid"].values).ffill().values
    return prev

def build(d):
    out = {}
    sid = d["sid"].values; t = d["t"].values; a = d["a"].values; typ = d["typ"].values; o = d["out"].values
    sids = pd.unique(sid)
    for direction, lab in [(1, "io"), (0, "oi")]:     # io: приход->расход, oi: расход->приход
        prev = _prev_opposite(d, direction)
        m = (o == direction) & ~np.isnan(prev)
        j = prev[m].astype(int)
        gap = (t[m] - t[j]).astype(float); dA = a[m] - a[j]
        pair = typ[j] * 4 + typ[m]
        big = a[j] > pd.Series(a[j]).groupby(sid[m]).transform(lambda s: s.quantile(0.75)).values
        df = pd.DataFrame({"sid": sid[m], "gap": gap, "dA": dA, "ad": np.abs(dA), "pair": pair, "big": big})
        g = df.groupby("sid")
        n = g.size()
        for sec, sl in [(300, "5m"), (3600, "1h"), (21600, "6h"), (86400, "24h")]:
            w = df["gap"] <= sec
            out[f"{lab}_share_{sl}"] = w.groupby(df["sid"]).mean()
            out[f"{lab}_sim_{sl}"] = (w & (df["ad"] < 0.05)).groupby(df["sid"]).mean()
        w24 = df[df["gap"] <= 86400]
        g24 = w24.groupby("sid")
        out[f"{lab}_dA_mean24"] = g24["dA"].mean()
        out[f"{lab}_ad_mean24"] = g24["ad"].mean()
        out[f"{lab}_ad_min"] = g["ad"].min()
        out[f"{lab}_ad_med"] = g["ad"].median()
        out[f"{lab}_gap_med"] = np.log1p(g["gap"].median())
        wb = df[df["big"] & (df["gap"] <= 86400)]
        out[f"{lab}_big_follow24"] = wb.groupby("sid").size() / n
        out[f"{lab}_big_sim24"] = (wb["ad"] < 0.1).groupby(wb["sid"]).mean()
        # пары по типам инструментов (в пределах суток), доли
        pc = pd.crosstab(w24["sid"], w24["pair"], normalize="index")
        for p in pc.columns:
            if (w24["pair"] == p).sum() > 20000:
                out[f"{lab}_pair_{TYPES[p//4][:4]}_{TYPES[p%4][:4]}"] = pc[p]
    F = pd.DataFrame(out).reindex(sids)
    return F.astype(np.float32)
