"""ФИНАЛЬНЫЙ ПАЙПЛАЙН: сырые файлы -> признаки -> 5 моделей -> ансамбль -> сабмит.
Запуск: python3 train_final.py            (~12 минут на 2 ядрах)
Всё строится с нуля, без кэшей. Seed'ы фиксированы.
"""
import json, time, numpy as np, pandas as pd, lightgbm as lgb
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostClassifier
from src.config import ID, TARGET, SUB_PRED, DATA, SUBMISSIONS
from src.data import load_signals, load_transactions
from src.features import build_features
from src.features_v3 import build_features_v3
from src.features_v4 import build_features_v4, CROSSES
from src.features_v7 import build_features_v7, type_norms
from src.features_v8 import build_features_v8
from src.hybrid import fit_hybrid

import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--seeds", default="42,202,777")
ap.add_argument("--tag", default="final")          # 'final' пишет сабмит
args = ap.parse_args()
T0 = time.time()
def log(m): print(f"[{time.time()-T0:6.0f}s] {m}", flush=True)
SEEDS = tuple(int(s) for s in args.seeds.split(","))
LGB_PARAMS = json.load(open("experiments/best_lgb.json"))

# ---------- 1. признаки ----------
def features(split, norms=None):
    sig = load_signals(split); tx = load_transactions(split)
    if norms is None:
        norms = type_norms(tx)                  # нормы инструментов — ТОЛЬКО по train
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
assert (Xtr.index.values == sig_tr[ID].values).all()
assert (Xte.index.values == sig_te[ID].values).all()
y = sig_tr.set_index(ID)[TARGET].reindex(Xtr.index)
# --- аудит: целостность и отсутствие таргета в признаках
assert y.notna().all() and len(y) == len(sig_tr)
assert TARGET not in Xtr.columns and TARGET not in Ltr.columns
assert list(Xtr.columns) == list(Xte.columns) and list(Ltr.columns) == list(Lte.columns)
assert Xtr.index.is_unique and Xte.index.is_unique and not set(Xtr.index) & set(Xte.index)
log(f"признаки: деревья {Xtr.shape[1]}, линейная {Ltr.shape[1]}; train {len(Xtr)}, test {len(Xte)}")

# заполнение пропусков для линейной модели: медианы и флаги — по train
lin_med = Ltr.median()
def lin_prep(L):
    na = pd.DataFrame({c + "_na": L[c].isna().astype(float) for c in Ltr.columns if Ltr[c].isna().any()}, index=L.index)
    return pd.concat([L.fillna(lin_med), na], axis=1)
Ztr, Zte = lin_prep(Ltr), lin_prep(Lte)

# ---------- 2. модели (одна функция = одна обученная модель на фолде) ----------
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
    full = np.concatenate([tr, es])            # линейной ранняя остановка не нужна
    m = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=5000)).fit(Ztr.iloc[full], y.iloc[full])
    return lambda Z: m.predict_proba(Z)[:, 1], "lin"

MODELS = {"lgbm": m_lgbm, "hybrid": m_hybrid,
          "cat_d4": m_cat(4, dict(l2_leaf_reg=20)),
          "cat_d6": m_cat(6, dict(l2_leaf_reg=30, rsm=0.5)),
          "linear": m_linear}

# ---------- 3. обучение: 3 seed x 5 фолдов, ранняя остановка на отдельной части ----------
per_seed = {k: {} for k in MODELS}   # name -> seed -> (oof, test)
oof = {k: np.zeros(len(y)) for k in MODELS}
test = {k: np.zeros(len(Xte)) for k in MODELS}
honest = {k: [] for k in MODELS}
for name, fn in MODELS.items():
    for seed in SEEDS:
        o = np.zeros(len(y)); tseed = np.zeros(len(Xte)); seen = np.zeros(len(y), int)
        for tr, va in StratifiedKFold(5, shuffle=True, random_state=seed).split(Xtr, y):
            tr2, es = train_test_split(tr, test_size=0.2, random_state=seed, stratify=y.iloc[tr])
            # аудит фолда: обучение / ранняя остановка / проверка не пересекаются
            assert not (set(tr2) & set(es)) and not (set(tr2) & set(va)) and not (set(es) & set(va))
            seen[va] += 1
            pred, kind = fn(tr2, es, seed)
            Xv, Xt = (Xtr.iloc[va], Xte) if kind == "tree" else (Ztr.iloc[va], Zte)
            o[va] = pred(Xv)
            pt = pred(Xt); tseed += pt / 5
            test[name] += pt / (5 * len(SEEDS))
        assert (seen == 1).all(), "каждая строка ровно один раз в проверке"
        per_seed[name][seed] = (o, tseed)
        honest[name].append(roc_auc_score(y, o))
        oof[name] += o / len(SEEDS)
    log(f"{name:7s} honest AUC = {np.mean(honest[name]):.5f} +- {np.std(honest[name]):.5f}")

# ---------- 4. ансамбль: правило объявлено ДО просмотра результатов ----------
sel = [k for k in MODELS if np.mean(honest[k]) > 0.640]
ens_oof = np.mean([rankdata(oof[k]) for k in sel], axis=0)
ens_te = np.mean([rankdata(test[k]) for k in sel], axis=0)
log(f"ансамбль {sel}: OOF AUC = {roc_auc_score(y, ens_oof):.5f}")

np.savez_compressed(f"experiments/preds_{args.tag}.npz",
    **{f"{k}__{sd}__oof": v[0] for k in per_seed for sd, v in per_seed[k].items()},
    **{f"{k}__{sd}__test": v[1] for k in per_seed for sd, v in per_seed[k].items()})
if args.tag != "final":
    log("режим аудита: сабмит не пишется"); raise SystemExit
# ---------- 5. сабмит ----------
p = (ens_te - ens_te.min()) / (ens_te.max() - ens_te.min()) * 0.9 + 0.05
sub = pd.read_csv(DATA / "sample_submission.csv")
assert (sub[ID].values == Xte.index.values).all(), "порядок ID!"
sub[SUB_PRED] = p
assert sub[ID].is_unique and sub[SUB_PRED].between(0, 1).all() and sub[SUB_PRED].notna().all()
SUBMISSIONS.mkdir(exist_ok=True)
sub.to_csv(SUBMISSIONS / "team_3B832E89.csv", index=False)
json.dump({"honest": {k: [float(v) for v in honest[k]] for k in honest}, "selected": sel,
           "ensemble_oof_auc": float(roc_auc_score(y, ens_oof))},
          open("experiments/final_scores.json", "w"), indent=1)
log(f"сохранено submissions/team_3B832E89.csv  ({len(sub)} строк)")
