"""П3: последовательность инструментов. Ключевая идея — отделить ПОРЯДОК
от СОСТАВА: липкость = P(тот же инструмент следующим) / сумма p_i^2.
Если генератор перемешивает операции независимо, липкость ~ 1 у всех."""
import numpy as np, pandas as pd
from .txcache import TYPES

def build(d, key="cr", K=8):
    sid = d["sid"].values; c = d[key].values.astype(int); t = d["t"].values
    same_sid = np.r_[False, sid[1:] == sid[:-1]]
    nxt_same = same_sid                                   # строка i имеет предыдущую в том же алерте
    prev_c = np.r_[-1, c[:-1]]
    df = pd.DataFrame({"sid": sid, "c": c, "pc": prev_c, "ok": nxt_same,
                       "gap": np.r_[0, np.diff(t)].astype(float)})
    df = df[df["ok"]]
    df["same"] = (df["c"] == df["pc"]).astype(np.int8)
    g = df.groupby("sid")
    comp = pd.crosstab(d["sid"], d[key], normalize="index")
    expected_same = (comp ** 2).sum(axis=1)
    out = {}
    out[f"{key}_stick"] = g["same"].mean() / expected_same
    out[f"{key}_switch_rate"] = 1 - g["same"].mean()
    # максимальная серия одного инструмента
    run_id = (~(df["same"].astype(bool))).groupby(df["sid"]).cumsum()
    runs = df.groupby([df["sid"], run_id]).size()
    out[f"{key}_maxrun"] = runs.groupby(level=0).max()
    out[f"{key}_meanrun"] = runs.groupby(level=0).mean()
    # интервал до смены инструмента vs без смены
    out[f"{key}_gap_switch_vs_stay"] = (df[df.same == 0].groupby("sid")["gap"].median()
                                       / (df[df.same == 1].groupby("sid")["gap"].median() + 1))
    # переходы A->B относительно ожидаемого по составу (lift)
    tr = pd.crosstab(df["sid"], df["pc"] * K + df["c"], normalize="index")
    for code in tr.columns:
        a_, b_ = code // K, code % K
        if a_ == b_:
            continue
        if (df["pc"] * K + df["c"] == code).sum() < 30000:
            continue
        exp = comp[a_] * comp[b_]
        out[f"{key}_lift_{a_}_{b_}"] = tr[code] / (exp + 1e-6)
    # энтропия переходов
    p = tr.values.clip(1e-12, 1)
    out[f"{key}_trans_entropy"] = pd.Series(-(p * np.log(p)).sum(axis=1), index=tr.index)
    return pd.DataFrame(out).astype(np.float32)
