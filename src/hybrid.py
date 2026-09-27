"""Гибрид: бустинг стартует с прогноза линейной модели (init_score).

Сигнал почти линеен по уровням инструментов (LR на 48 признаках = 0.640,
бустинг на 238 = 0.643). Деревья приближают линейную границу «лесенкой».
Здесь линейная модель ловит главный контраст, а деревья учат только
НЕЛИНЕЙНЫЕ ПОПРАВКИ поверх неё.

Всё внутри фолда: LR обучается на той же обучающей доле, что и бустинг.
"""
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

C8 = ["karta_kirim", "karta_chiqim", "bank_otkazmasi_kirim", "bank_otkazmasi_chiqim",
      "naqd_kirim", "naqd_chiqim", "xalqaro_kirim", "xalqaro_chiqim"]
LIN_COLS = [f"x_{c}_{s}" for c in C8 for s in ["mean", "median", "q25", "q75", "q90", "min"]]


def lin_matrix(X: pd.DataFrame, med: pd.Series = None):
    Z = X[LIN_COLS].copy()
    if med is None:
        med = Z.median()
    na = Z.isna().astype(np.float32).add_suffix("_na")
    na = na.loc[:, [c + "_na" for c in LIN_COLS if Z[c].isna().any() or c in med.index]]
    return pd.concat([Z.fillna(med), na], axis=1), med


def fit_hybrid(Xa, ya, Xes, yes, seed, params, lin_C=0.05, n_rounds=3000):
    """Возвращает функцию predict(X) -> вероятность."""
    Za, med = lin_matrix(Xa)
    lr = make_pipeline(StandardScaler(), LogisticRegression(C=lin_C, max_iter=5000)).fit(Za, ya)
    def lin_logit(X):
        Z, _ = lin_matrix(X, med)
        Z = Z.reindex(columns=Za.columns, fill_value=0)
        return lr.decision_function(Z)
    P = dict(params, objective="binary", metric="auc", bagging_freq=1,
             verbosity=-1, num_threads=2, seed=seed)
    dtr = lgb.Dataset(Xa, ya, init_score=lin_logit(Xa))
    des = lgb.Dataset(Xes, yes, init_score=lin_logit(Xes))
    m = lgb.train(P, dtr, n_rounds, valid_sets=[des],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
    def predict(X):
        raw = m.predict(X, num_iteration=m.best_iteration, raw_score=True) + lin_logit(X)
        return 1 / (1 + np.exp(-raw))
    return predict, m.best_iteration
