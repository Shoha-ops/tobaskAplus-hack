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


# ================= КОНТРОЛИРУЕМЫЙ ЭКСПЕРИМЕНТ: A (обучение на tr2, 64%) vs B (дообучение на tr, 80%) =================
# B: best_iteration находится на tr2/es ровно как в A, затем модель переобучается на ПОЛНОМ tr
#    с ФИКСИРОВАННЫМ числом итераций (без ранней остановки) и предсказывает va и test.
#    va в обоих вариантах в обучении не участвует. Фолды и seed'ы идентичны.
from src.hybrid import lin_matrix
def lgbP(seed): return dict(LGB_PARAMS, objective="binary", metric="auc", bagging_freq=1, verbosity=-1, num_threads=2, seed=seed)

def ab_lgbm(tr, tr2, es, seed):
    m = lgb.train(lgbP(seed), lgb.Dataset(Xtr.iloc[tr2], y.iloc[tr2]), 3000,
                  valid_sets=[lgb.Dataset(Xtr.iloc[es], y.iloc[es])], callbacks=[lgb.early_stopping(100, verbose=False)])
    it = m.best_iteration
    mB = lgb.train(lgbP(seed), lgb.Dataset(Xtr.iloc[tr], y.iloc[tr]), it)
    return (lambda X: m.predict(X, num_iteration=it)), (lambda X: mB.predict(X)), it

def ab_hybrid(tr, tr2, es, seed):
    prA, it = fit_hybrid(Xtr.iloc[tr2], y.iloc[tr2], Xtr.iloc[es], y.iloc[es], seed, LGB_PARAMS)
    Xa, ya = Xtr.iloc[tr], y.iloc[tr]                       # B: та же конструкция, но на полном tr
    Za, med = lin_matrix(Xa)
    lr = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=5000)).fit(Za, ya)
    def lin_logit(X):
        Z, _ = lin_matrix(X, med); return lr.decision_function(Z.reindex(columns=Za.columns, fill_value=0))
    mB = lgb.train(lgbP(seed), lgb.Dataset(Xa, ya, init_score=lin_logit(Xa)), max(it, 1))
    prB = lambda X: 1 / (1 + np.exp(-(mB.predict(X, raw_score=True) + lin_logit(X))))
    return prA, prB, it

def ab_cat(depth, extra):
    def f(tr, tr2, es, seed):
        kw = dict(depth=depth, learning_rate=0.02, random_seed=seed, verbose=0, thread_count=2, **extra)
        m = CatBoostClassifier(iterations=4000, eval_metric="AUC", early_stopping_rounds=150, **kw)
        m.fit(Xtr.iloc[tr2], y.iloc[tr2], eval_set=(Xtr.iloc[es], y.iloc[es]))
        it = m.get_best_iteration() + 1
        mB = CatBoostClassifier(iterations=it, **kw).fit(Xtr.iloc[tr], y.iloc[tr])
        return (lambda X: m.predict_proba(X)[:, 1]), (lambda X: mB.predict_proba(X)[:, 1]), it
    return f

def ab_linear(tr, tr2, es, seed):      # линейная и так учится на полном tr: A == B
    m = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=5000)).fit(Ztr.iloc[tr], y.iloc[tr])
    p = lambda Z: m.predict_proba(Z)[:, 1]
    return p, p, 0

MODELS = {"lgbm": (ab_lgbm, "tree"), "hybrid": (ab_hybrid, "tree"), "cat_d4": (ab_cat(4, dict(l2_leaf_reg=20)), "tree"),
          "cat_d6": (ab_cat(6, dict(l2_leaf_reg=30, rsm=0.5)), "tree"), "linear": (ab_linear, "lin")}
R = {}
for name, (fn, kind) in MODELS.items():
    for seed in SEEDS:
        oA = np.zeros(len(y)); oB = np.zeros(len(y)); tA = np.zeros(len(Xte)); tB = np.zeros(len(Xte)); its = []
        for tr, va in StratifiedKFold(5, shuffle=True, random_state=seed).split(Xtr, y):
            tr2, es = train_test_split(tr, test_size=0.2, random_state=seed, stratify=y.iloc[tr])
            assert not (set(tr) & set(va))
            pA, pB, it = fn(tr, tr2, es, seed); its.append(it)
            Xv, Xt = (Xtr.iloc[va], Xte) if kind == "tree" else (Ztr.iloc[va], Zte)
            oA[va] = pA(Xv); oB[va] = pB(Xv); tA += pA(Xt) / 5; tB += pB(Xt) / 5
        R[f"{name}__{seed}__A_oof"] = oA; R[f"{name}__{seed}__B_oof"] = oB
        R[f"{name}__{seed}__A_test"] = tA; R[f"{name}__{seed}__B_test"] = tB
        log(f"{name:7s} seed {seed}: A={roc_auc_score(y, oA):.5f}  B={roc_auc_score(y, oB):.5f}  Δ={roc_auc_score(y, oB)-roc_auc_score(y, oA):+.5f}  итераций≈{int(np.median(its))}")
R["y"] = y.values
np.savez_compressed("experiments/refit_AB.npz", **R)
log("готово")
