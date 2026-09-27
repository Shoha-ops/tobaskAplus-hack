"""Генератор: суммы систематически снижаются к дате алерта (+0.39 за 180д,
календарь ~0). Тренд общий -> случайные моменты операций добавляют шум в
уровни инструментов. Убираю ГЛОБАЛЬНЫЙ тренд g(дней до алерта), оценённый по
train БЕЗ таргета (среднее отклонение от нормы инструмента по дневным корзинам,
сглаженное), и применяю те же числа к test."""
import json, numpy as np, pandas as pd
from .config import DATA, EXPERIMENTS

def fit_trend(d):
    x = (d["sb"].values / 86400).clip(0, 181)
    dev = d["a"].values - d.groupby("cr")["a"].transform("mean").values
    bins = np.arange(0, 182, 3.0)
    idx = np.digitize(x, bins) - 1
    m = pd.Series(dev).groupby(idx).mean()
    c = (bins[:-1] + 1.5)[m.index.values.clip(0, len(bins) - 2)]
    # сглаживание полиномом 3 степени по центрам корзин (вес = число операций)
    w = pd.Series(dev).groupby(idx).size().values
    coef = np.polyfit(c, m.values, 3, w=np.sqrt(w))
    json.dump({"coef": coef.tolist()}, open(EXPERIMENTS / "trend.json", "w"))
    return coef

def apply_trend(d, coef):
    x = (d["sb"].values / 86400).clip(0, 181)
    d = d.copy(); d["a"] = d["a"].values - np.polyval(coef, x)
    return d


def detrended_transactions(split, coef):
    """Транзакции в исходном формате, но miqdor_indeksi без общего тренда."""
    from .data import load_signals, load_transactions
    from .config import ID, DATE
    sig = load_signals(split); tx = load_transactions(split)
    sb = (tx[ID].map(sig.set_index(ID)[DATE]) - tx["tranzaksiya_vaqti"]).dt.total_seconds().values
    x = (sb / 86400).clip(0, 181)
    tx["miqdor_indeksi"] = (tx["miqdor_indeksi"].values.astype(np.float64) - np.polyval(coef, x)).astype(np.float32)
    return sig, tx
