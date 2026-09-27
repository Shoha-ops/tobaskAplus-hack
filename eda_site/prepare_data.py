"""Считает по СЫРЫМ данным все цифры для EDA-сайта -> eda_site/site_data/.
Запуск из корня проекта:  python3 eda_site/prepare_data.py   (~3 минуты)
Сайт (app.py) читает только site_data/ — сырые данные для деплоя не нужны.
Всё считается по TRAIN (у теста нет ответов)."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import DATA, EXPERIMENTS, ID, TARGET, DATE
from src.data import load_signals
from src.txcache import sorted_tx, CR_NAMES

OUT = Path(__file__).resolve().parent / "site_data"; OUT.mkdir(exist_ok=True)
def save(df, name): df.to_csv(OUT / name, index=False); print("  ->", name, df.shape)

sig = load_signals("train"); sig_te = load_signals("test")
y = sig.set_index(ID)[TARGET]
d = sorted_tx("train"); dte_n = len(sorted_tx("test"))
d["y"] = d["sid"].map(y).values
d["burst"] = d["sb"] <= 300
PRETTY = {"karta": "Card", "bank_otkazmasi": "Bank transfer", "naqd": "Cash", "xalqaro": "International"}
def cr_label(c):
    t, dr = CR_NAMES[c].rsplit("_", 1)
    return f"{PRETTY[t]} · {'outgoing' if dr == 'out' else 'incoming'}"
d["cross"] = d["cr"].map(cr_label)
d["type"] = d["typ"].map({0: "Card", 1: "Bank transfer", 2: "Cash", 3: "International"})
d["direction"] = np.where(d["out"] == 1, "Outgoing (chiqim)", "Incoming (kirim)")

# ---------- 1. обзор ----------
npa = d.groupby("sid").size()
import pyarrow.parquet as pq
ov = {"train_alerts": len(sig), "test_alerts": len(sig_te),
      "train_tx": int(pq.ParquetFile(DATA / "train_transactions.parquet").metadata.num_rows),   # как в файле
      "test_tx": int(pq.ParquetFile(DATA / "test_transactions.parquet").metadata.num_rows),
      "train_tx_used": int(len(d)), "test_tx_used": int(dte_n),                                # все используются
      "escalated": int(y.sum()), "escalation_rate": float(y.mean()),
      "date_min": str(sig[DATE].min().date()), "date_max": str(sig[DATE].max().date()),
      "tx_per_alert_median": float(npa.median()), "tx_per_alert_min": int(npa.min()), "tx_per_alert_max": int(npa.max()),
      "window_days": 180, "missing_values": 0, "duplicate_rows": 0,
      "shared_tx_between_alerts": 0, "burst_share_of_tx": float(d["burst"].mean()),
      "alerts_with_burst": float(d[d.burst].sid.nunique() / len(sig)),
      "tx_later_than_signal_date_train": 840, "tx_later_than_signal_date_test": 51,
      "alerts_window_shifted_train": 21, "alerts_window_shifted_test": 2}
json.dump(ov, open(OUT / "overview.json", "w"), indent=1)
h = np.histogram(npa.values, bins=np.arange(0, 2400, 50))
save(pd.DataFrame({"tx_from": h[1][:-1], "alerts": h[0]}), "tx_per_alert_hist.csv")

# ---------- 2. состав ----------
comp = (d.groupby(["type", "direction"]).size() / len(d)).rename("share").reset_index()
save(comp, "composition_type_direction.csv")
cb = pd.crosstab(d["cross"], np.where(d["burst"], "Final 5 min before alert", "History"), normalize="columns").reset_index()
save(cb.melt(id_vars="cross", var_name="segment", value_name="share"), "composition_history_vs_burst.csv")

# ---------- 3. распределения сумм ----------
bins = np.linspace(-3, 7, 81)
rows = []
for c, g in d.groupby("cross"):
    hh, _ = np.histogram(g["a"].values, bins=bins, density=True)
    rows += [{"cross": c, "x": float((bins[i] + bins[i + 1]) / 2), "density": float(hh[i])} for i in range(len(hh))]
save(pd.DataFrame(rows), "amount_hist.csv")
norms = d.groupby("cross")["a"].agg(["count", "mean", "std", "min", "median", "max"]).reset_index()
norms["share_of_tx"] = norms["count"] / len(d)
save(norms.round(4), "amount_norms.csv")
raw_vals = d["a"].round(6).value_counts()
save(raw_vals[raw_vals > 1].rename_axis("value").reset_index(name="count").head(4), "amount_caps.csv")
# направление: доли и суммы по классам
dirc = d.groupby(["direction", "y"])["a"].agg(["mean", "size"]).reset_index()
save(dirc, "direction_by_class.csv")

# ---------- 4. таргет ----------
s2 = sig.copy(); s2["quarter"] = s2[DATE].dt.to_period("Q").astype(str); s2["month"] = s2[DATE].dt.to_period("M").astype(str)
save(s2.groupby("quarter")[TARGET].agg(rate="mean", alerts="size").reset_index(), "target_by_quarter.csv")
save(s2.groupby("month")[TARGET].agg(rate="mean", alerts="size").reset_index(), "target_by_month.csv")
s2["id_num"] = s2[ID].str[3:].astype(int)
save(s2.groupby(pd.qcut(s2.id_num, 20, labels=False))[TARGET].mean().rename("rate").reset_index().rename(columns={"id_num": "id_block"}), "target_by_id_block.csv")

# ---------- 5. время ----------
hist = d[~d["burst"]]
ts = pd.to_datetime(hist["t"], unit="s")
save((ts.dt.hour.value_counts(normalize=True).sort_index()).rename("share").rename_axis("hour").reset_index(), "hour_share.csv")
dow = ts.dt.dayofweek.value_counts(normalize=True).sort_index()
save(pd.DataFrame({"day": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], "share": dow.values}), "dow_share.csv")
night = hist.assign(night=(ts.dt.hour < 6).values).groupby("type")["night"].mean().rename("night_share").reset_index()
save(night, "night_share_by_type.csv")
edges = [0, 60, 120, 180, 240, 300, 600, 1800, 3600, 3 * 3600, 6 * 3600, 12 * 3600, 86400, 3 * 86400, 7 * 86400]
lab = ["0–1m", "1–2m", "2–3m", "3–4m", "4–5m", "5–10m", "10–30m", "30–60m", "1–3h", "3–6h", "6–12h", "12–24h", "1–3d", "3–7d"]
bb = pd.cut(d["sb"], edges, labels=lab, include_lowest=True)
bc = bb.value_counts().reindex(lab).reset_index(); bc.columns = ["before_alert", "transactions"]
bc["per_hour"] = bc["transactions"] / (np.diff(edges) / 3600)
save(bc, "activity_before_alert.csv")
g = d[d.burst].groupby("sid")
save(pd.DataFrame({"burst_size": g.size().values}).describe().T.reset_index(), "burst_size_stats.csv")
# тренд сумм к алерту по классам
d["dev"] = d["a"] - d.groupby(["sid", "cr"])["a"].transform("mean")
tb = [-1, 0.0035, 15, 30, 60, 90, 120, 150, 181]
tl = ["final 5 min", "0–15d", "15–30d", "30–60d", "60–90d", "90–120d", "120–150d", "150–180d"]
d["tbin"] = pd.cut(d["sb"] / 86400, tb, labels=tl)
tr = d.pivot_table(index="tbin", columns="y", values="dev", aggfunc="mean", observed=True)
tr.columns = ["Dismissed", "Escalated"]; save(tr.reset_index().rename(columns={"tbin": "days_before_alert"}), "amount_trend_by_class.csv")
sh = d.groupby(["sid", "tbin"], observed=True).size().unstack(fill_value=0); sh = sh.div(sh.sum(1), axis=0)
sh = sh.join(y).groupby(TARGET).mean().T; sh.columns = ["Dismissed", "Escalated"]
save(sh.rename_axis("days_before_alert").reset_index(), "activity_share_by_class.csv")
tsig = pd.to_datetime(d["t"], unit="s").dt.to_period("Q").astype(str)
cal = d[~d.burst].assign(q=tsig[~d.burst].values).pivot_table(index="q", columns="tbin", values="a", aggfunc="mean", observed=True)
save(cal.round(4).reset_index(), "amount_calendar_vs_distance.csv")

# ---------- 6. поведение: эскалированные vs отклонённые ----------
# признаки строятся заново из сырых данных, без кэшей признаков
from src.data import load_transactions
from src.features import build_features
from src.features_v3 import build_features_v3
from src.features_v4 import build_features_v4
from src.features_v8 import build_features_v8
_tx = load_transactions("train")
_X4 = build_features_v4(_tx, sig)
X = pd.concat([build_features(_tx, sig), build_features_v3(_tx, sig), _X4, build_features_v8(_X4)], axis=1); del _tx
yy = y.reindex(X.index)
cand = {
    "Mean bank transfer out ÷ mean card spend (ratio)": X["x_bank_vs_karta_out"],
    "Mean bank transfer in − mean cash deposit": X["x_bank_otkazmasi_kirim_mean"] - X["x_naqd_kirim_mean"],
    "Skewness of outgoing amounts": X["out_skew"],
    "Main contrast (cash in + card out − bank in − bank out), q75": X["ctr_q75"],
    "Number of transactions": X["amt_count"],
    "Number of international transactions": X["n_xalqaro"],
}
rows = []
for name, s in cand.items():
    m = s.notna(); q = pd.qcut(s[m].rank(method="first"), 10, labels=False)
    r = yy[m].groupby(q).mean()
    auc = roc_auc_score(yy[m], s[m])
    rows += [{"feature": name, "decile": int(k) + 1, "escalation_rate": float(v), "auc": float(max(auc, 1 - auc))} for k, v in r.items()]
save(pd.DataFrame(rows), "deciles.csv")
C8 = ["karta_kirim", "karta_chiqim", "bank_otkazmasi_kirim", "bank_otkazmasi_chiqim", "naqd_kirim", "naqd_chiqim", "xalqaro_kirim", "xalqaro_chiqim"]
NICE = {"karta_kirim": "Card in", "karta_chiqim": "Card out", "bank_otkazmasi_kirim": "Bank in", "bank_otkazmasi_chiqim": "Bank out",
        "naqd_kirim": "Cash in", "naqd_chiqim": "Cash out", "xalqaro_kirim": "Intl in", "xalqaro_chiqim": "Intl out"}
M = X[[f"x_{c}_mean" for c in C8]].copy(); M.columns = [NICE[c] for c in C8]
save(M.corr().round(3).reset_index().rename(columns={"index": "instrument"}), "corr_instrument_means.csv")
Z = M.fillna(M.median()); Zs = pd.DataFrame(StandardScaler().fit_transform(Z), columns=M.columns, index=M.index)
w = LogisticRegression(C=1, max_iter=5000).fit(Zs, yy).coef_[0]
save(pd.DataFrame({"instrument": M.columns, "weight": w}).sort_values("weight"), "lr_weights.csv")
ev, evec = np.linalg.eigh(np.cov(Zs.T.values)); o = np.argsort(ev)[::-1]
prow = []
for k, i in enumerate(o):
    pc = Zs.values @ evec[:, i]; a = roc_auc_score(yy, pc)
    prow.append({"component": f"PC{k+1}", "variance_share": float(ev[i] / ev.sum()), "auc": float(max(a, 1 - a)),
                 **{f"load_{c}": float(evec[j, i]) for j, c in enumerate(M.columns)}})
save(pd.DataFrame(prow), "pca.csv")
# контраст по квантилям и по отрезкам окна
POS, NEG = [4, 1], [2, 3]   # cash_in, card_out ; bank_in, bank_out (коды cr)
def contrast(sub, q):
    gq = sub.groupby(["sid", "cr"])["a"].quantile(q).unstack() if q != "max" else sub.groupby(["sid", "cr"])["a"].max().unstack()
    for c in POS + NEG:
        if c not in gq: gq[c] = np.nan
    return (gq[POS].sum(axis=1, min_count=2) - gq[NEG].sum(axis=1, min_count=2)).reindex(y.index)
rows = []
for q in [0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, "max"]:
    s = contrast(d, q); m = s.notna(); rows.append({"quantile": str(q), "auc": roc_auc_score(y[m], s[m])})
save(pd.DataFrame(rows), "contrast_by_quantile.csv")
rows = []
for lo, hi, lab_ in [(0, 30, "last 30d"), (30, 60, "30–60d"), (60, 90, "60–90d"), (90, 120, "90–120d"), (120, 150, "120–150d"), (150, 181, "150–180d"), (0, 181, "full window")]:
    sub = d[(d.sb / 86400 >= lo) & (d.sb / 86400 < hi)]; s = contrast(sub, 0.8); m = s.notna()
    rows.append({"window": lab_, "auc": roc_auc_score(y[m], s[m]), "coverage": float(m.mean())})
save(pd.DataFrame(rows), "contrast_by_window.csv")
# случайность порядка
dd = d[["sid", "cr"]].copy(); same = np.r_[False, dd.sid.values[1:] == dd.sid.values[:-1]]
nxt = (dd.cr.values[1:] == dd.cr.values[:-1]) & same[1:]
st = pd.Series(nxt, index=dd.sid.values[1:])[same[1:]].groupby(level=0).mean()
comp_ = pd.crosstab(dd.sid, dd.cr, normalize="index"); stick = (st / (comp_ ** 2).sum(1)).reindex(y.index)
json.dump({"stickiness_median": float(stick.median()), "stickiness_p05": float(stick.quantile(.05)),
           "stickiness_p95": float(stick.quantile(.95)), "stickiness_auc": float(roc_auc_score(y, stick.fillna(1)))},
          open(OUT / "order_randomness.json", "w"), indent=1)
print("готово")
