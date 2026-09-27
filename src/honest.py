"""Общая честная CV (ранняя остановка на отдельной части обучающей доли).
Возвращает усреднённый по seed'ам OOF и список AUC."""
import json, numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score

def honest_lgb(X, y, params=None, seeds=(42, 202, 777), verbose=True):
    if params is None:
        params = json.load(open("experiments/best_lgb.json"))
    P = dict(params, objective="binary", metric="auc", bagging_freq=1,
             verbosity=-1, num_threads=2)
    sc, oofs = [], []
    for seed in seeds:
        oof = np.zeros(len(y))
        for tr, va in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
            tr2, es = train_test_split(tr, test_size=0.2, random_state=seed, stratify=y.iloc[tr])
            m = lgb.train(dict(P, seed=seed), lgb.Dataset(X.iloc[tr2], y.iloc[tr2]), 3000,
                          valid_sets=[lgb.Dataset(X.iloc[es], y.iloc[es])],
                          callbacks=[lgb.early_stopping(100, verbose=False)])
            oof[va] = m.predict(X.iloc[va], num_iteration=m.best_iteration)
        sc.append(roc_auc_score(y, oof)); oofs.append(oof)
    if verbose:
        print(f"  honest AUC = {np.mean(sc):.5f} +- {np.std(sc):.5f}")
    return np.mean(oofs, axis=0), sc
