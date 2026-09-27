"""Признаки формы распределения сумм.

Мотивация (проверено вручную на train):
  - out_skew даёт чистую монотонную зависимость: 0.209 -> 0.111 по децилям.
    Асимметрия НЕ зависит от числа наблюдений -> признак честный.
  - amt_min сильно коррелирует с n_tx (-0.346): минимум это порядковая
    статистика, она механически падает с ростом N. Признак мутный, чиним.

Всё считается внутри одного signal_id. Глобальных статистик нет -> лика нет.
"""
import numpy as np
import pandas as pd
from .config import ID

QS = [0.01, 0.02, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.98, 0.99]


def _shape_block(g, prefix):
    """Характеристики формы, инвариантные к числу наблюдений."""
    q = g["miqdor_indeksi"].quantile(QS).unstack()
    q.columns = [f"q{int(c*100):02d}" for c in q.columns]

    f = pd.DataFrame(index=q.index)
    f[f"{prefix}_kurt"] = g["miqdor_indeksi"].apply(pd.Series.kurt)

    # размахи
    iqr = q["q75"] - q["q25"]
    f[f"{prefix}_iqr"] = iqr
    f[f"{prefix}_idr"] = q["q90"] - q["q10"]          # inter-decile range
    f[f"{prefix}_range99"] = q["q99"] - q["q01"]

    # асимметрия Боули: робастная версия skew, устойчива к выбросам
    f[f"{prefix}_bowley"] = (q["q75"] + q["q25"] - 2 * q["q50"]) / (iqr + 1e-6)
    # хвостовая асимметрия по децилям
    f[f"{prefix}_tail_asym"] = (q["q90"] + q["q10"] - 2 * q["q50"]) / (q["q90"] - q["q10"] + 1e-6)
    # квантильный эксцесс: насколько тяжёлые хвосты относительно середины
    f[f"{prefix}_qkurt"] = (q["q90"] - q["q10"]) / (iqr + 1e-6)
    f[f"{prefix}_qkurt99"] = (q["q99"] - q["q01"]) / (iqr + 1e-6)
    # какой хвост длиннее: левый или правый
    f[f"{prefix}_tail_ratio"] = (q["q50"] - q["q01"]) / (q["q99"] - q["q50"] + 1e-6)
    f[f"{prefix}_tail_ratio_d"] = (q["q50"] - q["q10"]) / (q["q90"] - q["q50"] + 1e-6)
    # коэффициент вариации через робастные оценки
    f[f"{prefix}_rcv"] = iqr / (q["q50"].abs() + 1e-6)

    # низкие квантили — спокойная замена минимуму (устойчивы к N)
    for c in ["q01", "q02", "q05"]:
        f[f"{prefix}_{c}"] = q[c]
    # насколько сам минимум оторван от остального хвоста
    f[f"{prefix}_min_gap"] = q["q01"] - g["miqdor_indeksi"].min()

    return f


def build_features_v3(tx: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
    tx = tx.copy()
    tx["is_out"] = (tx["kirim_chiqim"].astype(str) == "chiqim").astype(np.int8)
    idx = signals[ID]

    out = _shape_block(tx.groupby(ID, observed=True), "sh_amt")
    # расходы информативнее прихода -> считаем отдельно
    for flag, prefix in [(1, "sh_out"), (0, "sh_in")]:
        sub = tx[tx["is_out"] == flag]
        out = out.join(_shape_block(sub.groupby(ID, observed=True), prefix))

    # модуль суммы: отдельная характеристика "размаха" безотносительно знака
    tx["abs_amt"] = tx["miqdor_indeksi"].abs()
    ga = tx.groupby(ID, observed=True)["abs_amt"]
    out["abs_mean"] = ga.mean()
    out["abs_std"] = ga.std()
    out["abs_max"] = ga.max()
    out["abs_skew"] = ga.skew()

    return out.reindex(idx).astype(np.float32)
