import json, numpy as np, pandas as pd, time
from src.detrend import detrended_transactions
from src.features import build_features
from src.features_v3 import build_features_v3
from src.features_v4 import build_features_v4
from src.features_v8 import build_features_v8
from src.screen import control, cv_scores
from src.config import DATA, EXPERIMENTS
t0=time.time()
coef=np.array(json.load(open(EXPERIMENTS/"trend.json"))["coef"])
sig,tx=detrended_transactions("train",coef)
X1=build_features(tx,sig); X3=build_features_v3(tx,sig); X4=build_features_v4(tx,sig); del tx
XD=pd.concat([X1,X3,X4,build_features_v8(X4)],axis=1)
XD.to_parquet(DATA/"feat_train_v1348_dt.parquet"); X4.to_parquet(DATA/"feat_train_v4_dt.parquet")
print(f"набор на очищенных суммах: {XD.shape}  {time.time()-t0:.0f}s")
X,y,s0=control(); XD=XD[X.columns]
s1,oof=cv_scores(XD,y); d=s1-s0
print(f"LightGBM  исходный {s0.mean():.5f}   очищенный {s1.mean():.5f}   Δ={d.mean():+.5f} [по seed {' '.join(f'{v:+.4f}' for v in d)}]")
np.save(DATA/"oof_lgbm_dt.npy",oof)
