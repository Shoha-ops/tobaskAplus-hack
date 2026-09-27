"""Builds notebooks/solution.ipynb. Feature/model code is copied VERBATIM from src/
so the notebook runs exactly the same code as train_final.py."""
import json
import re
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
BEST = json.load(open(ROOT / "experiments" / "best_lgb.json"))
ST = json.load(open(ROOT / "experiments" / "stability_nofilter.json"))


def module(name):
    s = (SRC / f"{name}.py").read_text(encoding="utf-8")
    s = re.sub(r"^from \.config import .*\n", "", s, flags=re.M)
    s = re.sub(r"^from \.features_v4 import CROSSES\n", "", s, flags=re.M)
    return f"# ---- verbatim copy of src/{name}.py ----\n" + s.strip() + "\n"


md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = []
STAB = f"; single seed {ST['single_mean']:.4f} ± {ST['single_std']:.4f} (5 seeds)"
STAB5 = f"single-seed {ST['single_mean']:.4f} ± {ST['single_std']:.4f}, test-ranking Spearman {ST['spearman']:.3f}"
ADV = f"{ST['adversarial_auc']:.3f}"

cells.append(md(f"""# AI Financial Alert Risk Scoring — reproducible solution

Predicts the probability that each hidden-test alert is **escalated** (metric: ROC-AUC).
This notebook reproduces the full pipeline **from the raw competition files to the submission** in one pass.

| | |
|---|---|
| Final model | equal-weight rank ensemble of 5 models (CatBoost ×2, LightGBM, hybrid linear+boosting, logistic regression) |
| Validation | StratifiedKFold(5) × 3 seeds, early stopping on a separate 20 % slice of each training fold |
| CV ROC-AUC | **0.65334** (3-seed bag){STAB} |
| Runtime | ~7 min on 2 CPU cores (`FAST_MODE=True`: ~2.5 min, 1 seed, slightly different numbers) |
| EDA website | see the submission form (Streamlit app) |

**How to run.** Put the five competition files in `data/` (or the project root — found automatically), set `TEAM_ID`, then *Run All*.
The output is `submissions/team_<TEAM_ID>.csv`. Package versions used are printed in the first cell and pinned in `requirements.txt`."""))

cells.append(code(f"""import glob, json, sys, time, warnings
from pathlib import Path
import numpy as np, pandas as pd
import lightgbm as lgb, catboost, sklearn, scipy, pyarrow
warnings.filterwarnings("ignore")

TEAM_ID   = "3B832E89"                 # official team ID
# folder with the 5 competition files: first of these that contains train_signals.csv
DATA_DIR  = next(p for p in [Path("../data"), Path(".."), Path("data"), Path(".")] if (p / "train_signals.csv").exists())
OUT_DIR   = Path("../submissions")
FAST_MODE = False                      # True: 1 seed for a quick check (numbers differ slightly)
SEEDS     = (42,) if FAST_MODE else (42, 202, 777)

# column names
ID, DATE, TARGET, SUB_PRED = "signal_id", "signal_sanasi", "eskalatsiya", "ehtimollik"
DATA = DATA_DIR

# LightGBM parameters (tuned once with Optuna on honest CV; frozen)
LGB_PARAMS = {json.dumps(BEST, indent=1)}

print("data:", DATA_DIR)
print("python", sys.version.split()[0])
for m in [pd, np, sklearn, lgb, catboost, scipy, pyarrow]:
    print(f"{{m.__name__:12s}} {{m.__version__}}")
T0 = time.time()
def log(msg): print(f"[{{time.time()-T0:6.0f}}s] {{msg}}", flush=True)"""))

# ---------------------------------------------------------------- data
cells.append(md("""## 1. Data loading and integrity checks

The data are relational: one alert ↔ many transactions (median ~460 over 180 days).
`miqdor_indeksi` is cast to float32 to save memory (the final model was trained this way; it only affects exact
equality of amounts, which no feature uses).

**Leakage rule:** only information available at the alert moment may be used. Each alert's history is a 180-day
window that ends with a ~3-minute burst of transactions right before the alert. `signal_sanasi` is a date **without
time**: for 99 % of alerts the burst ends just before 00:00 of that date, while for 21 train / 2 test alerts the whole
window is shifted by one day and the burst ends just before 24:00 of the same date (840 / 51 transactions later than
00:00). Those transactions are the same pre-alert burst, not post-alert data, so **all transactions are used**
(removing them would delete these alerts' burst and shorten their window). The check below verifies this."""))
cells.append(code(module("data").replace(
    'def load_sample_submission() -> pd.DataFrame:\n    return pd.read_csv(DATA / "sample_submission.csv")',
    'def load_sample_submission() -> pd.DataFrame:\n'
    '    f = sorted(glob.glob(str(DATA / "sample_submission*.csv")))[0]   # tolerate "sample_submission (3).csv"\n'
    '    return pd.read_csv(f)')))
cells.append(code("""sig_tr, sig_te = load_signals("train"), load_signals("test")
sub_tpl = load_sample_submission()
print(f"train alerts {len(sig_tr):,}   test alerts {len(sig_te):,}   escalation rate {sig_tr[TARGET].mean():.4f}")

# --- integrity checks (fail loudly if anything is off)
assert sig_tr[ID].is_unique and sig_te[ID].is_unique
assert not set(sig_tr[ID]) & set(sig_te[ID]), "train/test alert IDs overlap"
assert (sub_tpl[ID].values == sig_te[ID].values).all(), "sample_submission order != test_signals order"
for split, sig in [("train", sig_tr), ("test", sig_te)]:
    raw = load_transactions(split)
    sb = (raw[ID].map(sig.set_index(ID)[DATE]) - raw["tranzaksiya_vaqti"]).dt.total_seconds()
    later = raw.loc[sb < 0, ID].unique()
    span = sb.groupby(raw[ID]).agg(lambda v: v.max() - v.min()) / 86400
    r64 = pd.read_parquet(DATA / f"{split}_transactions.parquet", columns=[ID, "tranzaksiya_vaqti", "miqdor_indeksi"])  # float64: float32 creates false ties
    dup = r64.duplicated(["tranzaksiya_vaqti", "miqdor_indeksi"], keep=False)
    shared = r64.loc[dup].groupby(["tranzaksiya_vaqti", "miqdor_indeksi"])[ID].nunique().gt(1).sum()
    del r64
    print(f"{split}: {len(raw):,} tx | later than 00:00 of signal date: {(sb < 0).sum()} tx in {len(later)} alerts, "
          f"all within that date: {(sb >= -86400).all()} | max history span {span.max():.3f} days "
          f"| every alert has history: {raw[ID].nunique() == len(sig)} | tx shared between alerts: {shared}")
    assert (sb >= -86400).all() and span.max() <= 180.0 + 1e-6 and raw[ID].nunique() == len(sig) and shared == 0
    del raw"""))

# ---------------------------------------------------------------- EDA
cells.append(md("""## 2. EDA highlights that shaped the model

The full EDA is on the team website. Three findings drove the modelling decisions:

1. **Each instrument has its own 'normal' amount** (card-in ≈ −0.5, international ≈ +2) → never average instruments
   together; compute statistics per *instrument × direction* (largest gain, +0.039 AUC).
2. **A common customer scale hides the signal.** Mean amounts are strongly correlated across instruments; the
   predictive part is a small contrast: *cash deposits and card spending large relative to bank transfers*.
3. **Time carries no signal** (uniform timestamps, random instrument order, identical trend in both classes);
   the ~3-minute burst before each alert is real but uninformative."""))
cells.append(code("""import matplotlib.pyplot as plt
tx = load_transactions("train")
cr = tx["tranzaksiya_turi"].astype(str) + "_" + tx["kirim_chiqim"].astype(str)
means = tx.groupby([tx[ID], cr], observed=True)["miqdor_indeksi"].mean().unstack()
y_all = sig_tr.set_index(ID)[TARGET].reindex(means.index)
contrast = means["naqd_kirim"] + means["karta_chiqim"] - means["bank_otkazmasi_kirim"] - means["bank_otkazmasi_chiqim"]
sb = (tx[ID].map(sig_tr.set_index(ID)[DATE]) - tx["tranzaksiya_vaqti"]).dt.total_seconds()

fig, ax = plt.subplots(1, 3, figsize=(17, 4.2))
order = ["karta_kirim", "karta_chiqim", "bank_otkazmasi_kirim", "bank_otkazmasi_chiqim", "naqd_kirim", "naqd_chiqim"]
ax[0].boxplot([tx.loc[cr == c, "miqdor_indeksi"].sample(20000, random_state=0) for c in order], showfliers=False)
ax[0].set_xticks(range(1, 7), [c.replace("bank_otkazmasi", "bank").replace("_", "\\n") for c in order])
ax[0].set_title("Amount by instrument × direction"); ax[0].set_ylabel("miqdor_indeksi")
m = contrast.notna(); q = pd.qcut(contrast[m].rank(method="first"), 10, labels=False)
ax[1].bar(range(1, 11), y_all[m].groupby(q).mean().values, color="#eb6834")
ax[1].axhline(y_all.mean(), ls=":", c="gray"); ax[1].set_title("Escalation rate by decile of the contrast")
ax[1].set_xlabel("decile (cash-in + card-out − bank-in − bank-out)")
bins = [0, 60, 120, 180, 300, 3600, 86400, 7 * 86400]
cnt = pd.cut(sb, bins, include_lowest=True).value_counts().sort_index()
ax[2].bar(["0-1m", "1-2m", "2-3m", "3-5m", "5m-1h", "1h-1d", "1-7d"], cnt.values / (np.diff(bins) / 3600), color="#2a78d6")
ax[2].set_yscale("log"); ax[2].set_title("Transactions per hour before the alert (log)")
plt.tight_layout(); plt.show()
from sklearn.metrics import roc_auc_score
print(f"Single hand-written contrast (no training): ROC-AUC = {roc_auc_score(y_all[m], contrast[m]):.4f}")
del tx, cr, sb"""))

# ---------------------------------------------------------------- features
cells.append(md("""## 3. Features (one row per alert, computed only from that alert's own pre-alert history)

| group | # | description |
|---|---|---|
| base aggregates | 59 | amount stats (all / incoming / outgoing), type shares & counts, balance, activity, alert date |
| distribution shape | 49 | kurtosis, Bowley skew, inter-quantile ranges, tail ratios |
| instrument × direction | 130 | count / mean / std / quantiles for each of 8 combinations — **the key EDA insight** |
| explicit contrast | 13 | cash-in + card-out − bank-in − bank-out at several statistics |
| linear-model inputs | 76 | instrument levels + customer-scale / skew decomposition + log counts |

No global statistics use the target. Instrument norms and missing-value medians are computed on **train only**
and applied unchanged to test."""))
for name in ["features", "features_v3", "features_v4", "features_v7", "features_v8"]:
    cells.append(code(module(name)))
cells.append(code("""def features(split, norms=None):
    sig = load_signals(split); tx = load_transactions(split)
    if norms is None:
        norms = type_norms(tx)                  # instrument norms - TRAIN only
    X1 = build_features(tx, sig)
    X3 = build_features_v3(tx, sig)
    X4 = build_features_v4(tx, sig)
    del tx
    X7 = build_features_v7(X4, norms)
    X8 = build_features_v8(X4)
    tree = pd.concat([X1, X3, X4, X8], axis=1)
    lin_cols = [f"x_{c}_{s}" for c in CROSSES for s in ["mean", "median", "q25", "q75", "q90", "min"]]
    dec = [c for c in X7.columns if not c.startswith("d_")]
    cnt = np.log1p(X4[[f"x_{c}_count" for c in CROSSES]].fillna(0)).add_suffix("_log")
    lin = pd.concat([X4[lin_cols], X7[dec], cnt], axis=1)
    return sig, tree, lin, norms

sig_tr, Xtr, Ltr, norms = features("train")
sig_te, Xte, Lte, _ = features("test", norms)
Xte = Xte[Xtr.columns]; Lte = Lte[Ltr.columns]
y = sig_tr.set_index(ID)[TARGET].reindex(Xtr.index)

assert (Xtr.index.values == sig_tr[ID].values).all() and (Xte.index.values == sig_te[ID].values).all()
assert y.notna().all() and TARGET not in Xtr.columns and TARGET not in Ltr.columns
assert list(Xtr.columns) == list(Xte.columns) and list(Ltr.columns) == list(Lte.columns)

lin_med = Ltr.median()                          # train-only statistics for the linear model
def lin_prep(L):
    na = pd.DataFrame({c + "_na": L[c].isna().astype(float) for c in Ltr.columns if Ltr[c].isna().any()}, index=L.index)
    return pd.concat([L.fillna(lin_med), na], axis=1)
Ztr, Zte = lin_prep(Ltr), lin_prep(Lte)
log(f"features: trees {Xtr.shape[1]}, linear {Ltr.shape[1]}; train {len(Xtr)}, test {len(Xte)}")"""))

# ---------------------------------------------------------------- models
cells.append(md("""## 4. Models and validation

* **Validation:** train/test are split at random (identical date ranges, interleaved IDs) and every alert is a separate
  customer (no shared transactions), so `StratifiedKFold(5)` is appropriate. A single split varied by ±0.015 AUC, so
  every result is averaged over 3 seeds.
* **Honest early stopping:** inside each training fold 20 % is held out *only* for early stopping; the validation fold
  never influences training (using it inflated AUC by +0.005).
* **Models:** the main signal is almost linear in instrument levels, while trees approximate it as a staircase — hence a
  logistic regression and a **hybrid** (LightGBM boosted from the logistic-regression score) next to CatBoost/LightGBM."""))
cells.append(code(module("hybrid")))
cells.append(code("""from scipy.stats import rankdata
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostClassifier

def m_lgbm(tr, es, seed):
    P = dict(LGB_PARAMS, objective="binary", metric="auc", bagging_freq=1, verbosity=-1, num_threads=2, seed=seed)
    m = lgb.train(P, lgb.Dataset(Xtr.iloc[tr], y.iloc[tr]), 3000,
                  valid_sets=[lgb.Dataset(Xtr.iloc[es], y.iloc[es])],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
    return lambda X: m.predict(X, num_iteration=m.best_iteration), "tree"

def m_hybrid(tr, es, seed):
    pr, _ = fit_hybrid(Xtr.iloc[tr], y.iloc[tr], Xtr.iloc[es], y.iloc[es], seed, LGB_PARAMS)
    return pr, "tree"

def m_cat(depth, extra):
    def f(tr, es, seed):
        m = CatBoostClassifier(iterations=4000, depth=depth, learning_rate=0.02, random_seed=seed,
                               eval_metric="AUC", verbose=0, thread_count=2, early_stopping_rounds=150, **extra)
        m.fit(Xtr.iloc[tr], y.iloc[tr], eval_set=(Xtr.iloc[es], y.iloc[es]))
        return lambda X: m.predict_proba(X)[:, 1], "tree"
    return f

def m_linear(tr, es, seed):
    full = np.concatenate([tr, es])            # no early stopping needed
    m = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=5000)).fit(Ztr.iloc[full], y.iloc[full])
    return lambda Z: m.predict_proba(Z)[:, 1], "lin"

MODELS = {"lgbm": m_lgbm, "hybrid": m_hybrid,
          "cat_d4": m_cat(4, dict(l2_leaf_reg=20)),
          "cat_d6": m_cat(6, dict(l2_leaf_reg=30, rsm=0.5)),
          "linear": m_linear}

oof = {k: np.zeros(len(y)) for k in MODELS}
test = {k: np.zeros(len(Xte)) for k in MODELS}
honest = {k: [] for k in MODELS}
for name, fn in MODELS.items():
    for seed in SEEDS:
        o = np.zeros(len(y)); seen = np.zeros(len(y), int)
        for tr, va in StratifiedKFold(5, shuffle=True, random_state=seed).split(Xtr, y):
            tr2, es = train_test_split(tr, test_size=0.2, random_state=seed, stratify=y.iloc[tr])
            assert not (set(tr2) & set(es)) and not (set(tr2) & set(va)) and not (set(es) & set(va))
            seen[va] += 1
            pred, kind = fn(tr2, es, seed)
            Xv, Xt = (Xtr.iloc[va], Xte) if kind == "tree" else (Ztr.iloc[va], Zte)
            o[va] = pred(Xv)
            test[name] += pred(Xt) / (5 * len(SEEDS))
        assert (seen == 1).all()
        honest[name].append(roc_auc_score(y, o))
        oof[name] += o / len(SEEDS)
    log(f"{name:7s} honest AUC = {np.mean(honest[name]):.5f} +- {np.std(honest[name]):.5f}")"""))

# ---------------------------------------------------------------- ensemble + submission
cells.append(md("""## 5. Ensemble

Rule fixed **before** looking at blend results: equal weights (rank average) for every model with honest AUC > 0.640.
Picking the best of many blend combinations would tune the ensemble to the CV."""))
cells.append(code("""sel = [k for k in MODELS if np.mean(honest[k]) > 0.640]
ens_oof = np.mean([rankdata(oof[k]) for k in sel], axis=0)
ens_te = np.mean([rankdata(test[k]) for k in sel], axis=0)
print(pd.DataFrame({k: [np.mean(v), np.std(v)] for k, v in honest.items()}, index=["AUC mean", "std"]).T.round(5))
log(f"ENSEMBLE {sel}: OOF ROC-AUC = {roc_auc_score(y, ens_oof):.5f}")"""))
cells.append(md("""## 6. Submission

ROC-AUC depends only on ranking; ranks are mapped linearly to [0.05, 0.95] to give valid probabilities in [0, 1]."""))
cells.append(code("""p = (ens_te - ens_te.min()) / (ens_te.max() - ens_te.min()) * 0.9 + 0.05
sub = load_sample_submission()
assert (sub[ID].values == Xte.index.values).all()
sub[SUB_PRED] = p
OUT_DIR.mkdir(exist_ok=True)
path = OUT_DIR / f"team_{TEAM_ID}.csv"
sub.to_csv(path, index=False)

chk = pd.read_csv(path); raw = open(path).read().splitlines()
checks = {
    "columns == ['signal_id', 'ehtimollik']": list(chk.columns) == ["signal_id", "ehtimollik"],
    "6000 rows": len(chk) == len(sig_te) == 6000,
    "order == test_signals": (chk[ID].values == sig_te[ID].values).all(),
    "no duplicate IDs": chk[ID].is_unique,
    "no missing / unknown IDs": set(chk[ID]) == set(sig_te[ID]),
    "no NaN / inf": np.isfinite(chk[SUB_PRED]).all(),
    "0 <= p <= 1": chk[SUB_PRED].between(0, 1).all(),
    "no index column": raw[0] == "signal_id,ehtimollik",
}
for k, v in checks.items(): print(("OK  " if v else "FAIL"), k)
assert all(checks.values())
log(f"saved {path}")
chk.head()"""))
cells.append(md(f"""## 7. Appendix — what we tested and rejected, and how we audited ourselves

Rule: an idea is kept only if it adds ≥ +0.001 AUC consistently across seeds (paired comparison on identical folds).

| idea | Δ AUC | why it failed |
|---|---|---|
| time windows 7/30/90 d, bursts, night activity, gaps | −0.0006 | timestamps are uniform |
| time / shape features per instrument | −0.0017 / −0.0015 | signal is in location, not timing or shape |
| amount histogram; 148 pairwise ratios | −0.0012 / −0.0002 | redundant with existing features |
| transaction-level (multiple-instance) model | alone 0.556 | a single transaction cannot see cross-instrument ratios |
| incoming → outgoing chains; instrument sequences | −0.0034 / −0.0009 | instrument order is random given composition |
| structuring (near-equal amounts) | +0.0003 (unstable) | amounts are continuous, no repeats |
| amounts detrended toward the alert | −0.011 | raw minima (late / burst transactions) carry legitimate pre-alert information |

**Audit.** Shuffled labels → 0.48–0.51 (no leakage in the pipeline); the key contrast is re-selected identically on two
disjoint halves; adversarial validation train vs test AUC {ADV}; 8 fresh 10k/4k hold-outs give a selection-bias estimate of
≈0.0035 ± 0.0036; the ensemble is stable across 5 seeds ({STAB5}).
Using ⅛ … all of each customer's transactions, the achievable AUC for this feature family extrapolates to ≈0.652–0.657."""))

nb = nbf.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
    "language_info": {"name": "python"}})
out = ROOT / "notebooks" / "solution.ipynb"
nbf.write(nb, out)
print("written", out, len(cells), "cells")
