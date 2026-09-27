"""Валидация. Train/test разбиты случайно (даты совпадают, ID перемешаны),
группировать нечего: каждый signal_id уникален. -> StratifiedKFold."""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from .config import SEED, N_FOLDS


def get_folds(y):
    return StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)


def cv_lgbm(X, y, params=None, num_boost_round=3000, verbose=False):
    """5-fold CV LightGBM. Возвращает (oof, scores, models)."""
    import lightgbm as lgb

    base = dict(
        objective="binary", metric="auc", learning_rate=0.03,
        num_leaves=31, min_child_samples=40, feature_fraction=0.8,
        bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
        seed=SEED, verbosity=-1, num_threads=2,
    )
    if params:
        base.update(params)

    oof = np.zeros(len(y))
    scores, models = [], []
    for fold, (tr, va) in enumerate(get_folds(y).split(X, y)):
        dtr = lgb.Dataset(X.iloc[tr], y.iloc[tr])
        dva = lgb.Dataset(X.iloc[va], y.iloc[va])
        m = lgb.train(base, dtr, num_boost_round=num_boost_round,
                      valid_sets=[dva],
                      callbacks=[lgb.early_stopping(150, verbose=False),
                                 lgb.log_evaluation(200 if verbose else 0)])
        oof[va] = m.predict(X.iloc[va], num_iteration=m.best_iteration)
        s = roc_auc_score(y.iloc[va], oof[va])
        scores.append(s)
        models.append(m)
        print(f"  fold {fold}: AUC={s:.5f}  best_iter={m.best_iteration}")
    print(f"  OOF AUC = {roc_auc_score(y, oof):.5f} | "
          f"mean={np.mean(scores):.5f} std={np.std(scores):.5f}")
    return oof, scores, models


def log_experiment(exp_id, features, model, scores, params, notes=""):
    import json, datetime
    from .config import EXPERIMENTS
    EXPERIMENTS.mkdir(exist_ok=True)
    f = EXPERIMENTS / "results.csv"
    row = pd.DataFrame([{
        "exp_id": exp_id, "features": features, "model": model,
        "cv_mean": float(np.mean(scores)), "cv_std": float(np.std(scores)),
        "folds": json.dumps([round(s, 5) for s in scores]),
        "params": json.dumps(params or {}),
        "notes": notes,
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
    }])
    row.to_csv(f, mode="a", header=not f.exists(), index=False)
    return row


def repeated_cv_lgbm(X, y, params=None, seeds=(42, 202, 777), quiet=True):
    """Повторная CV по нескольким seed'ам. Нужна потому, что разброс
    между фолдами (~0.015 AUC) больше, чем эффекты, которые мы измеряем."""
    import lightgbm as lgb
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score

    base = dict(objective="binary", metric="auc", learning_rate=0.02,
                num_leaves=7, min_child_samples=150, feature_fraction=0.5,
                bagging_fraction=0.7, bagging_freq=1, lambda_l2=20.0,
                verbosity=-1, num_threads=2)
    if params:
        base.update(params)

    all_scores, oofs = [], []
    for seed in seeds:
        p = dict(base, seed=seed)
        oof = np.zeros(len(y))
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        for tr, va in skf.split(X, y):
            dtr = lgb.Dataset(X.iloc[tr], y.iloc[tr])
            dva = lgb.Dataset(X.iloc[va], y.iloc[va])
            m = lgb.train(p, dtr, num_boost_round=3000, valid_sets=[dva],
                          callbacks=[lgb.early_stopping(150, verbose=False)])
            oof[va] = m.predict(X.iloc[va], num_iteration=m.best_iteration)
        s = roc_auc_score(y, oof)
        all_scores.append(s)
        oofs.append(oof)
        if not quiet:
            print(f"  seed {seed}: OOF AUC = {s:.5f}")
    mean, std = float(np.mean(all_scores)), float(np.std(all_scores))
    print(f"  repeated-CV AUC = {mean:.5f} +- {std:.5f}  (seeds={list(seeds)})")
    return np.mean(oofs, axis=0), all_scores
