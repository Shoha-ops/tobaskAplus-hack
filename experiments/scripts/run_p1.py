import time, numpy as np, pandas as pd
from src.txcache import sorted_tx
from src.hyp_pairs import build
from src.screen import screen
from src.config import DATA
from sklearn.metrics import roc_auc_score
from src.screen import control
t0=time.time()
F=build(sorted_tx("train")); F.to_parquet(DATA/"hyp_p1.parquet")
print(f"П1: {F.shape[1]} признаков  {time.time()-t0:.0f}s")
X,y,_=control(); F=F.reindex(X.index)
print("одиночные AUC (топ-8, >0.5 или <0.5):")
au={c:roc_auc_score(y[F[c].notna()],F[c].dropna()) for c in F.columns}
for c,v in sorted(au.items(),key=lambda t:-abs(t[1]-0.5))[:8]: print(f"   {c:30s} {v:.4f}")
screen("P1_pairs_all",F,"приход<->расход: окна, близость сумм, пары типов")
screen("P1_pairs_io_only",F[[c for c in F if c.startswith("io_")]],"только приход->расход")
