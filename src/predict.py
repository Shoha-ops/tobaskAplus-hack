"""Финальное обучение и сабмит. Конфигурация заморожена по результатам CV."""
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from .config import DATA, SUBMISSIONS, ID, TARGET, SUB_PRED, N_FOLDS

PARAMS = dict(objective="binary", metric="auc", learning_rate=0.02, num_leaves=7,
              min_child_samples=150, feature_fraction=0.5, bagging_fraction=0.7,
              bagging_freq=1, lambda_l2=20.0, verbosity=-1, num_threads=2)


def fit_predict(Xtr, y, Xte, seeds=(42, 202, 777), params=None):
    """Bagging по seed'ам + по фолдам. Возвращает (oof, test_pred)."""
    p = dict(PARAMS, **(params or {}))
    oofs, preds = [], []
    for seed in seeds:
        oof = np.zeros(len(y)); pr = np.zeros(len(Xte))
        skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=seed)
        for tr, va in skf.split(Xtr, y):
            m = lgb.train(dict(p, seed=seed), lgb.Dataset(Xtr.iloc[tr], y.iloc[tr]), 3000,
                          valid_sets=[lgb.Dataset(Xtr.iloc[va], y.iloc[va])],
                          callbacks=[lgb.early_stopping(150, verbose=False)])
            oof[va] = m.predict(Xtr.iloc[va], num_iteration=m.best_iteration)
            pr += m.predict(Xte, num_iteration=m.best_iteration) / N_FOLDS
        print(f"  seed {seed}: OOF AUC = {roc_auc_score(y, oof):.5f}")
        oofs.append(oof); preds.append(pr)
    return np.mean(oofs, axis=0), np.mean(preds, axis=0)


def make_submission(test_ids, pred, name):
    SUBMISSIONS.mkdir(exist_ok=True)
    sub = pd.read_csv(DATA / "sample_submission.csv")
    assert (sub[ID].values == np.asarray(test_ids)).all(), "порядок строк не совпадает!"
    sub[SUB_PRED] = np.clip(pred, 0.0, 1.0)
    path = SUBMISSIONS / name
    sub.to_csv(path, index=False)
    print(f"saved {path}  n={len(sub)}  range=[{sub[SUB_PRED].min():.5f}, {sub[SUB_PRED].max():.5f}]")
    return sub
