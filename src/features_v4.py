"""Суммы в разрезе ТИП x НАПРАВЛЕНИЕ + гистограмма распределения сумм.

Мотивация (проверено вручную по децилям на train):
  mean bank_otkazmasi_chiqim: 0.230 -> 0.113  (размах 0.117)
  mean xalqaro_chiqim:        0.237 -> 0.144  (размах 0.093)
  mean bank_otkazmasi_kirim:  0.220 -> 0.126  (размах 0.094)
  для сравнения out_skew:     0.209 -> 0.111  (размах 0.098)

То есть средняя сумма банковского перевода сильнее прежнего чемпиона.
Статистики сумм В РАЗРЕЗЕ ТИПА модель раньше не видела вообще: были только
доли типов и статистики сумм по направлению, но не их пересечение.

Границы гистограммы фиксированные (заданы константой), а не посчитанные
по данным -> одинаковы для train и test, лика нет.
"""
import numpy as np
import pandas as pd
from .config import ID

CROSSES = [
    "karta_kirim", "karta_chiqim",
    "bank_otkazmasi_kirim", "bank_otkazmasi_chiqim",
    "naqd_kirim", "naqd_chiqim",
    "xalqaro_kirim", "xalqaro_chiqim",
]

# фиксированные границы: miqdor_indeksi стандартизован организаторами,
# диапазон -2.91 .. 6.69 на train
BINS = np.array([-np.inf, -2.5, -2.0, -1.6, -1.2, -0.9, -0.6, -0.35, -0.1,
                 0.15, 0.4, 0.7, 1.0, 1.4, 1.8, 2.3, 3.0, 4.0, np.inf])


def build_features_v4(tx: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
    tx = tx.copy()
    tx["cross"] = (tx["tranzaksiya_turi"].astype(str) + "_"
                   + tx["kirim_chiqim"].astype(str))
    idx = signals[ID]
    out = pd.DataFrame(index=pd.Index(idx, name=ID))

    # --- статистики сумм по каждой из 8 комбинаций
    for c in CROSSES:
        sub = tx[tx["cross"] == c]
        if len(sub) == 0:
            continue
        g = sub.groupby(ID, observed=True)["miqdor_indeksi"]
        f = g.agg(["count", "mean", "std", "median", "max", "min", "sum", "skew"])
        f.columns = [f"x_{c}_{k}" for k in f.columns]
        q = g.quantile([0.25, 0.75, 0.90]).unstack()
        q.columns = [f"x_{c}_q{int(v*100):02d}" for v in q.columns]
        out = out.join(f.join(q))

    # --- отношения средних между инструментами:
    # "перевод в среднем во сколько раз крупнее карточной операции"
    def ratio(a, b, name):
        ca, cb = f"x_{a}_mean", f"x_{b}_mean"
        if ca in out.columns and cb in out.columns:
            out[name] = out[ca] / (out[cb].abs() + 1e-6)

    ratio("bank_otkazmasi_chiqim", "karta_chiqim", "x_bank_vs_karta_out")
    ratio("bank_otkazmasi_kirim", "karta_kirim", "x_bank_vs_karta_in")
    ratio("naqd_chiqim", "karta_chiqim", "x_naqd_vs_karta_out")
    ratio("xalqaro_chiqim", "karta_chiqim", "x_xalqaro_vs_karta_out")
    ratio("bank_otkazmasi_chiqim", "bank_otkazmasi_kirim", "x_bank_out_vs_in")
    ratio("naqd_chiqim", "naqd_kirim", "x_naqd_out_vs_in")

    # --- гистограмма сумм: доля транзакций в каждой корзине.
    # Моменты (mean/skew/kurt) сжимают форму с потерями; гистограмма
    # показывает её как есть, включая горбы и провалы.
    codes = np.digitize(tx["miqdor_indeksi"].values, BINS[1:-1])
    h = pd.crosstab(tx[ID].values, codes, normalize="index")
    h.columns = [f"hist_{int(c):02d}" for c in h.columns]
    out = out.join(h.reindex(idx))

    # --- гистограмма отдельно по расходам (расходы информативнее)
    sub = tx[tx["kirim_chiqim"].astype(str) == "chiqim"]
    codes = np.digitize(sub["miqdor_indeksi"].values, BINS[1:-1])
    ho = pd.crosstab(sub[ID].values, codes, normalize="index")
    ho.columns = [f"histout_{int(c):02d}" for c in ho.columns]
    out = out.join(ho.reindex(idx))

    return out.astype(np.float32)
