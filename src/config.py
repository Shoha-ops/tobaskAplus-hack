"""Глобальная конфигурация. Метрика подтверждена организаторами: ROC-AUC."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
EXPERIMENTS = ROOT / "experiments"
SUBMISSIONS = ROOT / "submissions"

SEED = 42
N_FOLDS = 5
METRIC = "roc_auc"          # подтверждено на странице соревнования

TARGET = "eskalatsiya"
ID = "signal_id"
DATE = "signal_sanasi"
SUB_PRED = "ehtimollik"     # имя колонки предсказаний в сабмите

WINDOW_DAYS = 180           # длина окна транзакций до даты алерта
