import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from src.txcache import sorted_tx
from src.hyp_seq import build
from src.screen import screen, control
from src.config import DATA
d=sorted_tx("train")
X,y,_=control()
F=pd.concat([build(d,"cr",8),build(d,"typ",4),build(d,"out",2)],axis=1).reindex(X.index)
F.to_parquet(DATA/"hyp_p3.parquet")
print("ПРОВЕРКА ГЕНЕРАТОРА: липкость (1.0 = порядок случаен при данном составе)")
for k in ["cr","typ","out"]:
    s=F[f"{k}_stick"]; print(f"   {k:4s}: медиана {s.median():.4f}  5%-95% [{s.quantile(.05):.4f}, {s.quantile(.95):.4f}]   AUC={roc_auc_score(y,s.fillna(s.median())):.4f}")
# эталон: та же статистика на ПЕРЕМЕШАННОМ порядке внутри каждого алерта
rng=np.random.RandomState(0); dd=d.copy()
dd["r"]=rng.rand(len(dd)); dd=dd.sort_values(["sid","r"]).reset_index(drop=True)
Fs=build(dd,"cr",8).reindex(X.index)["cr_stick"]
print(f"   эталон (порядок перемешан вручную): медиана {Fs.median():.4f}  5%-95% [{Fs.quantile(.05):.4f}, {Fs.quantile(.95):.4f}]")
au={c:roc_auc_score(y,F[c].fillna(F[c].median())) for c in F}
print("одиночные AUC топ-6:"); [print(f"   {c:28s} {v:.4f}") for c,v in sorted(au.items(),key=lambda t:-abs(t[1]-.5))[:6]]
screen("P3_sequence_all",F,"липкость, серии, lift переходов, энтропия; cr/typ/out")
