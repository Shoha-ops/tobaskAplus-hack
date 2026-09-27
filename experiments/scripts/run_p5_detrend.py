import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from src.txcache import sorted_tx, CR_NAMES
from src.detrend import fit_trend, apply_trend
from src.screen import control, screen
from src.config import DATA
d=sorted_tx("train"); X,y,_=control()
coef=fit_trend(d); print("тренд g(дни до алерта):"," ".join(f"{x}д:{np.polyval(coef,x):+.3f}" for x in [0,30,60,90,120,150,180]))
dd=apply_trend(d,coef)
def loc(df,pref):
    g=df.groupby(["sid","cr"]).a
    a=g.agg(["mean","median","min"]); q=g.quantile([.25,.75,.9]).unstack(); q.columns=["q25","q75","q90"]
    t=a.join(q).unstack("cr"); t.columns=[f"{pref}_{CR_NAMES[c]}_{s}" for s,c in t.columns]; return t.reindex(X.index)
def lr_auc(D):
    Z=D.copy(); na={c+"_na":Z[c].isna().astype(float) for c in Z if Z[c].isna().any()}
    Z=pd.concat([Z.fillna(Z.median()),pd.DataFrame(na,index=Z.index)],axis=1); sc=[]
    for s in (42,202,777):
        o=np.zeros(len(y))
        for a_,b_ in StratifiedKFold(5,shuffle=True,random_state=s).split(Z,y):
            o[b_]=make_pipeline(StandardScaler(),LogisticRegression(C=0.05,max_iter=5000)).fit(Z.iloc[a_],y.iloc[a_]).predict_proba(Z.iloc[b_])[:,1]
        sc.append(roc_auc_score(y,o))
    return np.mean(sc),np.array(sc)
L0=loc(d,"raw"); L1=loc(dd,"dt")
m0,s0=lr_auc(L0); m1,s1=lr_auc(L1)
print(f"\nЛинейная на уровнях инструментов:  сырые {m0:.5f}   без тренда {m1:.5f}   Δ={m1-m0:+.5f} [по seed {' '.join(f'{v:+.4f}' for v in s1-s0)}]")
def ctr(L,p,st):
    return L[f"{p}_naqd_in_{st}"]+L[f"{p}_karta_out_{st}"]-L[f"{p}_bank_otkazmasi_in_{st}"]-L[f"{p}_bank_otkazmasi_out_{st}"]
for st in ["mean","q75"]:
    a0=ctr(L0,"raw",st); a1=ctr(L1,"dt",st); m=a0.notna()&a1.notna()
    print(f"контраст {st:4s}: сырые {roc_auc_score(y[m],a0[m]):.4f}   без тренда {roc_auc_score(y[m],a1[m]):.4f}")
L1.to_parquet(DATA/"hyp_p5_detrended_loc.parquet")
screen("P5_detrended_levels",L1,"уровни инструментов после удаления общего тренда по дням до алерта")
