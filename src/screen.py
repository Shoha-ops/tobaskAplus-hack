"""Фильтр гипотез: честная CV LightGBM, 3 seed, ПАРНОЕ сравнение с контролем
на одинаковых фолдах. Журнал -> experiments/hypotheses.csv"""
import json, datetime, numpy as np, pandas as pd, lightgbm as lgb
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score
from .config import DATA, ID, TARGET, EXPERIMENTS
from .data import load_signals

SEEDS = (42, 202, 777)
ENSEMBLE_REF = 0.6533
_P = None

def params():
    global _P
    if _P is None:
        _P = dict(json.load(open(EXPERIMENTS / "best_lgb.json")), objective="binary",
                  metric="auc", bagging_freq=1, verbosity=-1, num_threads=2)
    return _P

def cv_scores(X, y, seeds=SEEDS):
    sc, oofs = [], []
    for seed in seeds:
        o = np.zeros(len(y))
        for tr, va in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
            tr2, es = train_test_split(tr, test_size=0.2, random_state=seed, stratify=y.iloc[tr])
            m = lgb.train(dict(params(), seed=seed), lgb.Dataset(X.iloc[tr2], y.iloc[tr2]), 3000,
                          valid_sets=[lgb.Dataset(X.iloc[es], y.iloc[es])],
                          callbacks=[lgb.early_stopping(100, verbose=False)])
            o[va] = m.predict(X.iloc[va], num_iteration=m.best_iteration)
        sc.append(roc_auc_score(y, o)); oofs.append(o)
    return np.array(sc), np.mean(oofs, axis=0)

def control():
    X = pd.read_parquet(DATA / "feat_train_v1348.parquet")
    y = load_signals("train").set_index(ID)[TARGET].reindex(X.index)
    f = EXPERIMENTS / "control_scores.json"
    if f.exists():
        sc = np.array(json.load(open(f))["scores"])
    else:
        sc, oof = cv_scores(X, y); json.dump({"scores": sc.tolist()}, open(f, "w"))
        np.save(DATA / "oof_control_lgbm.npy", oof)
    return X, y, sc

def screen(name, F, notes=""):
    X, y, s0 = control()
    F = F.reindex(X.index)
    s1, oof = cv_scores(X.join(F), y)
    d = s1 - s0
    verdict = ("НЕ БРАТЬ" if d.mean() < 0.001 else
               "перепроверить" if d.mean() < 0.002 else
               "ПРОВЕРИТЬ ЛИК" if d.mean() > 0.005 else "кандидат")
    if d.mean() >= 0.001 and (d > 0).sum() < 3:
        verdict += " (нестабильно)"
    print(f"  {name:34s} +{F.shape[1]:3d} призн.  AUC={s1.mean():.5f}±{s1.std():.5f}  "
          f"Δ={d.mean():+.5f} [по seed: {' '.join(f'{v:+.4f}' for v in d)}]  -> {verdict}", flush=True)
    row = pd.DataFrame([{"experiment": name, "features": F.shape[1], "model": "lgbm_honest_paired",
                         "AUC_mean": round(s1.mean(), 5), "AUC_std": round(s1.std(), 5),
                         "delta_vs_control": round(d.mean(), 5),
                         "delta_by_seed": " ".join(f"{v:+.4f}" for v in d),
                         "delta_vs_0.6533_est": round(d.mean(), 5),
                         "verdict": verdict, "notes": notes,
                         "time": datetime.datetime.now().isoformat(timespec="minutes")}])
    p = EXPERIMENTS / "hypotheses.csv"
    row.to_csv(p, mode="a", header=not p.exists(), index=False)
    return d.mean(), oof
