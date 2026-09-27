"""Явный контраст, найденный разведкой.

Средние по 8 инструментам сильно коррелированы (0.5–0.88): есть общий
«масштаб клиента» (PC1, 62% дисперсии), но он почти не предсказывает (AUC 0.535).
Сигнал — в малом контрасте (PC5, 4% дисперсии, AUC 0.617):
    + naqd_kirim + karta_chiqim  − bank_otkazmasi_kirim − bank_otkazmasi_chiqim
Деревья делят по отдельным средним, где доминирует бесполезный масштаб, и
контраст ловят плохо. Даю его явно. Веса ФИКСИРОВАНЫ (±1), по данным не
подгоняются -> лика нет.
"""
import numpy as np
import pandas as pd

POS = ["naqd_kirim", "karta_chiqim"]
NEG = ["bank_otkazmasi_kirim", "bank_otkazmasi_chiqim"]


def build_features_v8(X4: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=X4.index)
    for st in ["mean", "median", "q25", "q75"]:
        p = sum(X4[f"x_{c}_{st}"] for c in POS)
        n = sum(X4[f"x_{c}_{st}"] for c in NEG)
        out[f"ctr_{st}"] = p - n
        # две половины контраста отдельно
        out[f"ctr_{st}_cash_vs_bank_in"] = X4[f"x_naqd_kirim_{st}"] - X4[f"x_bank_otkazmasi_kirim_{st}"]
        out[f"ctr_{st}_card_vs_bank_out"] = X4[f"x_karta_chiqim_{st}"] - X4[f"x_bank_otkazmasi_chiqim_{st}"]
    # версия, устойчивая к пропуску наличных: только карта vs банк
    out["ctr_mean_nocash"] = (X4["x_karta_chiqim_mean"] + X4["x_karta_kirim_mean"]
                              - X4["x_bank_otkazmasi_kirim_mean"] - X4["x_bank_otkazmasi_chiqim_mean"])
    return out.astype(np.float32)
