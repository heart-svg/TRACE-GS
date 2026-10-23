#!/usr/bin/env python3
"""Leakage-controlled baselines on fold-specific S2G representations."""
import argparse, json, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.base import clone
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.ensemble import ExtraTreesRegressor


def pcc(y,p):
    return float(pearsonr(np.asarray(y,float),np.asarray(p,float)).statistic) if np.std(p)>0 else 0.0


def metrics(y,p):
    y,p=np.asarray(y,float),np.asarray(p,float)
    return {'PCC':pcc(y,p),'RMSE':float(np.sqrt(np.mean((y-p)**2))),'MAE':float(np.mean(np.abs(y-p)))}


def load_features(path, expected_ids):
    d=pd.read_csv(path,dtype={'ID':str}) if str(path).endswith(('.csv','.csv.gz')) else pd.read_parquet(path)
    idc='ID' if 'ID' in d.columns else d.columns[0]
    d[idc]=d[idc].astype(str)
    if d[idc].duplicated().any(): raise AssertionError(f'duplicate feature IDs: {path}')
    d=d.set_index(idc)
    if set(d.index)!=set(expected_ids): raise AssertionError(f'feature/manifest ID mismatch: {path}')
    d=d.drop(columns=['y_raw'],errors='ignore')
    x=d.loc[expected_ids].to_numpy(np.float32)
    if not np.isfinite(x).all(): raise AssertionError(f'nonfinite features: {path}')
    return x


def candidates(name,seed):
    if name=='ridge': return [(Ridge(alpha=a),{'alpha':a}) for a in (1.,10.,100.,1000.,10000.)]
    if name=='lasso': return [(make_pipeline(StandardScaler(),Lasso(alpha=a,max_iter=20000)),{'alpha':a}) for a in (.001,.01,.1,1.)]
    if name=='elasticnet': return [(make_pipeline(StandardScaler(),ElasticNet(alpha=a,l1_ratio=l,max_iter=20000)),{'alpha':a,'l1_ratio':l}) for a in (.001,.01,.1) for l in (.2,.5,.8)]
    if name=='svr': return [(make_pipeline(StandardScaler(),SVR(C=c,gamma=g,epsilon=.1)),{'C':c,'gamma':g}) for c in (0.1,1.,10.) for g in ('scale',.001)]
    if name=='extratrees': return [(ExtraTreesRegressor(n_estimators=500,max_features=m,min_samples_leaf=l,n_jobs=4,random_state=seed),{'max_features':m,'min_samples_leaf':l}) for m in ('sqrt',.2) for l in (1,3,5)]
    if name=='xgboost':
        import xgboost as xgb
        return [(xgb.XGBRegressor(n_estimators=800,learning_rate=.03,max_depth=d,
                 subsample=.8,colsample_bytree=.8,reg_lambda=lam,n_jobs=4,
                 random_state=seed,tree_method='hist'),{'max_depth':d,'reg_lambda':lam})
                for d in (3,6) for lam in (1.,10.)]
    raise ValueError(name)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--manifest',required=True,type=Path)
    ap.add_argument('--feature-template',required=True,help='format fields: {outer}, {split}')
    ap.add_argument('--model',required=True,choices=['ridge','lasso','elasticnet','svr','extratrees','xgboost'])
    ap.add_argument('--out',required=True,type=Path); ap.add_argument('--seed',type=int,default=20260902)
    a=ap.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
    man=pd.read_csv(a.manifest,dtype={'ID':str}); reports=[]; pooled=[]; start=time.time()
    for outer in range(5):
        parts={s:man[(man.outer==outer)&(man.split==s)].copy() for s in ('trn','val','tst')}
        if any(x.empty for x in parts.values()): raise AssertionError(f'missing split outer {outer}')
        X={s:load_features(Path(a.feature_template.format(outer=outer,split=s)),parts[s].ID.tolist()) for s in parts}
        scored=[]
        for est,hp in candidates(a.model,a.seed+outer):
            est.fit(X['trn'],parts['trn'].y_raw); pv=est.predict(X['val'])
            scored.append((float(np.mean((parts['val'].y_raw-pv)**2)),est,hp))
        _,best,hp=min(scored,key=lambda z:z[0])
        # Keep the model fitted on the training split. Validation labels select
        # hyperparameters only, exactly like validation-based early stopping in
        # TRACE/DEM; merging validation back into training would give baselines
        # extra labelled samples and make the comparison asymmetric.
        pt=best.predict(X['tst']); met=metrics(parts['tst'].y_raw,pt)
        report={'outer':outer,'model':a.model,'selected':hp,**met,
                'n_train':len(parts['trn']),'n_val':len(parts['val']),'n_test':len(pt),
                'validation_policy':'select_only_no_refit'}; reports.append(report)
        z=pd.DataFrame({'ID':parts['tst'].ID,'outer':outer,'y_raw':parts['tst'].y_raw,'pred_raw':pt}); pooled.append(z)
        od=a.out/f'outer_{outer}'; od.mkdir(exist_ok=True); z.to_csv(od/'predictions_tst.csv',index=False); (od/'metrics.json').write_text(json.dumps(report,indent=2))
        print('OUTER_PASS',json.dumps(report),flush=True)
    tab=pd.DataFrame(reports); pred=pd.concat(pooled,ignore_index=True)
    if pred.ID.duplicated().any(): raise AssertionError('pooled test IDs are not unique')
    agg={'status':'PASS','model':a.model,'folds':5,'PCC_mean':float(tab.PCC.mean()),'PCC_sd':float(tab.PCC.std(ddof=1)),
         'RMSE_mean':float(tab.RMSE.mean()),'MAE_mean':float(tab.MAE.mean()),'pooled':metrics(pred.y_raw,pred.pred_raw),'elapsed_seconds':time.time()-start}
    tab.to_csv(a.out/'fivefold_metrics.csv',index=False); pred.to_csv(a.out/'oof_predictions.csv',index=False)
    (a.out/'COMPLETE.json').write_text(json.dumps(agg,indent=2)); print('BASELINE_5FOLD_PASS',json.dumps(agg),flush=True)
if __name__=='__main__': main()

