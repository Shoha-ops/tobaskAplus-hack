"""П2: дробление — близкие по размеру суммы. Сырые float64 (float32 давал
ложные совпадения). Точных повторов в данных нет (кроме 4 значений-потолков),
поэтому только близость в пределах tol по лог-шкале.
Считаю отдельно: все / приход / расход / каждая из 8 комбинаций тип x направление."""
import numpy as np, pandas as pd
import pyarrow.parquet as pq
from .txcache import CR_NAMES
from .config import DATA

TOLS = [0.005, 0.02, 0.05]

def _near(sid, a, t, prefix):
    o = np.lexsort((a, sid)); s, v, tt = sid[o], a[o], t[o]
    same = np.r_[False, s[1:] == s[:-1]]
    dl = np.r_[np.inf, np.diff(v)]; dl[~same] = np.inf
    dr = np.r_[dl[1:], np.inf]
    nn = np.minimum(dl, dr)                                 # до ближайшего соседа по сумме
    df = pd.DataFrame({"sid": s, "nn": nn})
    out = {}
    g = df.groupby("sid")["nn"]
    for tol in TOLS:
        out[f"{prefix}_near{tol}"] = (df["nn"] <= tol).groupby(df["sid"]).mean()
    out[f"{prefix}_nn_med"] = np.log(g.median() + 1e-9)
    # крупнейшая группа в окне ширины 0.05 (через скользящее окно по отсортированным)
    grp = []
    for key, idx in pd.Series(np.arange(len(s))).groupby(s).indices.items():
        vv = v[idx]; j = np.searchsorted(vv, vv + 0.05, side="right") - np.arange(len(vv))
        grp.append((key, j.max(), j.max() / len(vv)))
    G = pd.DataFrame(grp, columns=["sid", "mx", "mxs"]).set_index("sid")
    out[f"{prefix}_maxgrp05"] = G["mx"]; out[f"{prefix}_maxgrp05_share"] = G["mxs"]
    return out

def build(split):
    raw = pq.read_table(DATA / f"{split}_transactions.parquet").to_pandas()
    sid = raw["signal_id"].astype(str).values; a = raw["miqdor_indeksi"].values.astype(np.float64)
    t = raw["tranzaksiya_vaqti"].values.astype("datetime64[s]").astype(np.int64)
    cr = (raw["tranzaksiya_turi"].astype(str) + "_" + raw["kirim_chiqim"].astype(str)).values
    out = {}
    out.update(_near(sid, a, t, "st_all"))
    for dname in ["kirim", "chiqim"]:
        m = raw["kirim_chiqim"].astype(str).values == dname
        out.update(_near(sid[m], a[m], t[m], f"st_{dname}"))
    for c in ["karta_kirim", "karta_chiqim", "bank_otkazmasi_kirim", "bank_otkazmasi_chiqim", "naqd_kirim", "naqd_chiqim"]:
        m = cr == c
        o = _near(sid[m], a[m], t[m], f"st_{c}")
        out.update({k: v for k, v in o.items() if "near0.02" in k or "maxgrp05_share" in k})
    return pd.DataFrame(out).replace([np.inf, -np.inf], np.nan).astype(np.float32)
