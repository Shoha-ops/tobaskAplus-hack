"""Отсортированные по времени транзакции с числовыми кодами (кэш)."""
import numpy as np, pandas as pd
from .config import DATA, ID, DATE
from .data import load_signals, load_transactions

CROSS = ["karta_kirim","karta_chiqim","bank_otkazmasi_kirim","bank_otkazmasi_chiqim",
         "naqd_kirim","naqd_chiqim","xalqaro_kirim","xalqaro_chiqim"]
TYPES = ["karta","bank_otkazmasi","naqd","xalqaro"]

def sorted_tx(split):
    f = DATA / f"txs_{split}.parquet"
    if f.exists():
        return pd.read_parquet(f)
    sig = load_signals(split); tx = load_transactions(split)
    d = pd.DataFrame({
        "sid": tx[ID].astype(str).values,
        "t": tx["tranzaksiya_vaqti"].values.astype("datetime64[s]").astype(np.int64),
        "out": (tx["kirim_chiqim"].astype(str) == "chiqim").values.astype(np.int8),
        "typ": tx["tranzaksiya_turi"].astype(str).map({t: i for i, t in enumerate(TYPES)}).values.astype(np.int8),
        "a": tx["miqdor_indeksi"].values.astype(np.float64)})
    d["cr"] = (d["typ"] * 2 + d["out"]).astype(np.int8)     # 0..7: karta_in, karta_out, bank_in, ...
    sd = sig.set_index(ID)[DATE].values.astype("datetime64[s]").astype(np.int64)
    d["sb"] = pd.Series(d["sid"]).map(pd.Series(sd, index=sig[ID].values)).values - d["t"]
    d = d.sort_values(["sid", "t"], kind="mergesort").reset_index(drop=True)
    d.to_parquet(f)
    return d

CR_NAMES = [f"{t}_{'out' if o else 'in'}" for t in TYPES for o in (0, 1)]
