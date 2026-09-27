import time, pandas as pd
from sklearn.metrics import roc_auc_score
from src.hyp_struct import build
from src.screen import screen, control
from src.config import DATA
t0=time.time(); X,y,_=control()
F=build("train").reindex(X.index); F.to_parquet(DATA/"hyp_p2.parquet")
print(f"П2: {F.shape[1]} признаков  {time.time()-t0:.0f}s")
au={c:roc_auc_score(y,F[c].fillna(F[c].median())) for c in F}
print("одиночные AUC топ-6:"); [print(f"   {c:34s} {v:.4f}") for c,v in sorted(au.items(),key=lambda t:-abs(t[1]-.5))[:6]]
print("корреляция топ-признака с числом транзакций:",round(F[max(au,key=lambda c:abs(au[c]-.5))].corr(X['amt_count']),3))
screen("P2_structuring_all",F,"близкие суммы +-tol, макс. группа, по направлениям и 6 инструментам; float64")
