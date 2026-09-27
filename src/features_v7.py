"""Разности расположения сумм между инструментами + разложение на
«масштаб клиента» и «перекос по инструментам».

Наблюдение: miqdor_indeksi ведёт себя как стандартизованный логарифм суммы
(у каждого инструмента свой сдвиг: karta_kirim ~ -0.54, xalqaro ~ +2.17).
В лог-шкале РАЗНОСТЬ = логарифм отношения реальных сумм. Проверка по децилям:
  mean(bank_kirim) - mean(naqd_kirim): 0.281 -> 0.104, размах 0.177
  прежний топ (отношение bank/karta):   размах 0.132
Отношения на стандартизованной шкале плохо ведут себя около нуля, разности нет.

Разложение: сумма ~ масштаб клиента + норма инструмента + перекос.
Норма инструмента = средняя по train-транзакциям (константа, таргет не нужен),
считается один раз по обучающим данным и применяется к тесту без изменений.
"""
import itertools
import numpy as np
import pandas as pd
from .config import ID
from .features_v4 import CROSSES

STATS = ["mean", "median", "q25", "q75"]


def type_norms(tx_train: pd.DataFrame) -> dict:
    """Нормы инструментов по ОБУЧАЮЩИМ транзакциям. Таргет не используется."""
    c = tx_train["tranzaksiya_turi"].astype(str) + "_" + tx_train["kirim_chiqim"].astype(str)
    g = tx_train["miqdor_indeksi"].astype(np.float64).groupby(c)
    return {"mean": g.mean().to_dict(), "median": g.median().to_dict()}


def build_features_v7(X4: pd.DataFrame, norms: dict) -> pd.DataFrame:
    """На вход — признаки v4 (статистики по 8 комбинациям), на выход — разности."""
    cols = {}

    # --- все попарные разности расположения
    for st in STATS:
        for a, b in itertools.combinations(CROSSES, 2):
            ca, cb = f"x_{a}_{st}", f"x_{b}_{st}"
            if ca in X4 and cb in X4:
                cols[f"d_{st}_{a}__{b}"] = X4[ca] - X4[cb]

    # --- разложение: отклонение от нормы инструмента
    for st in ["mean", "median"]:
        dev = pd.DataFrame({c: X4[f"x_{c}_{st}"] - norms[st][c]
                            for c in CROSSES if f"x_{c}_{st}" in X4})
        w = pd.DataFrame({c: X4[f"x_{c}_count"].fillna(0) for c in dev.columns})
        # масштаб клиента: взвешенное среднее отклонений по всем инструментам
        scale = (dev.fillna(0) * w).sum(axis=1) / w.sum(axis=1).clip(lower=1)
        cols[f"scale_{st}"] = scale
        # перекос: насколько каждый инструмент отличается от общего масштаба клиента
        for c in dev.columns:
            cols[f"skewto_{st}_{c}"] = dev[c] - scale
        cols[f"skewto_{st}_spread"] = dev.sub(scale, axis=0).std(axis=1)

    return pd.DataFrame(cols, index=X4.index).astype(np.float32)
