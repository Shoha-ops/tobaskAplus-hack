"""Data loading with memory-efficient types (essential: constrained to ~3 GB RAM)."""
import numpy as np
import pandas as pd
from .config import DATA, ID, DATE, TARGET


def load_signals(split: str) -> pd.DataFrame:
    df = pd.read_csv(DATA / f"{split}_signals.csv", parse_dates=[DATE])
    return df


def load_transactions(split: str) -> pd.DataFrame:
    df = pd.read_parquet(DATA / f"{split}_transactions.parquet")
    df["kirim_chiqim"] = df["kirim_chiqim"].astype("category")
    df["tranzaksiya_turi"] = df["tranzaksiya_turi"].astype("category")
    df["miqdor_indeksi"] = df["miqdor_indeksi"].astype(np.float32)
    return df


def load_sample_submission() -> pd.DataFrame:
    return pd.read_csv(DATA / "sample_submission.csv")
