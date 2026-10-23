"""Leakage-safe RNA residual fusion for formal stickleback five-fold evaluation."""
import json,os
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

base=Path(os.environ['STICKLEBACK_ROOT']); form=base/'formal'; inner=form/'strict_inner_s2g_v1'; final=form/'outer_final_s2g_v1'; out=form/'strict_rna_fusion_v1'; out.mkdir(exist_ok=True)
rna=pd.read_csv(form/'rna_gene_counts.csv.gz',dtype={'sample_id':str}).set_index('sample_id'); X=np.log1p(rna.to_numpy(float)/np.maximum(rna.to_numpy(float).sum(1,keepdims=True),1)*1e6); rids=rna.index.astype(str); ridx={s:i for i,s in enumerate(rids)}
spl=pd.read_csv(form/'nested_5x3_family_grouped_splits.csv',dtype={'sample_id':str}); testall=pd.read_csv(final/'trace_outer_oof_predictions.csv',dtype={'sample_id':str}); alphas=[.01,.1,1.,10.,100.]; reports=[]; preds=[]; nullrows=[]
def corr(a,b): return float(pearsonr(a,b).statistic) if len(a)>2 and np.std(a)>0 and np.std(b)>0 else float('nan')
def train_predict(tr_ids,te_ids,target,q,alpha):
 A=X[[ridx[s] for s in tr_ids]]; B=X[[ridx[s] for s in te_ids]]; vt=VarianceThreshold(0.); A=vt.fit_transform(A); B=vt.transform(B); sc=StandardScaler(); A=sc.fit_transform(A); B=sc.transform(B); k=min(q,len(tr_ids)-1,A.shape[1]); pc=PCA(n_components=k,svd_solver='randomized',random_state=20261009); A=pc.fit_transform(A); B=pc.transform(B); m=Ridge(alpha=alpha); m.fit(A,target); return m.predict(B),k
for outer in range(5):
 oof=pd.read_csv(inner/f'outer_{outer}/strict_oof_predictions.csv',dtype={'sample_id':str}); test=testall[testall.outer==outer].copy(); assert oof.sample_id.is_unique and test.sample_id.is_unique and not set(oof.sample_id)&set(test.sample_id); assert set(oof.sample_id)|set(test.sample_id)==set(rids)
 residual=oof.residual.to_numpy(float); sm=spl[(spl.outer==outer)&(spl.partition=='outer_train')].set_index('sample_id'); groups=sm.loc[oof.sample_id,'inner_holdout'].astype(int).to_numpy(); candidate={a:np.empty(len(oof)) for a in alphas}
 for h in range(3):
  va=np.flatnonzero(groups==h); tr=np.flatnonzero(groups!=h)
  for alpha in alphas: candidate[alpha][va],_=train_predict(oof.sample_id.iloc[tr].tolist(),oof.sample_id.iloc[va].tolist(),residual[tr],32,alpha)
 alpha=min(alphas,key=lambda z:float(np.mean((residual-candidate[z])**2))); ro=candidate[alpha]; gamma=float(min(np.linspace(0,1.5,151),key=lambda g:float(np.mean((residual-g*ro)**2))))
 rt,q=train_predict(oof.sample_id.tolist(),test.sample_id.tolist(),residual,32,alpha); y=test.observed.to_numpy(float); dna=test.prediction.to_numpy(float); fused=dna+gamma*rt; rng=np.random.default_rng(20261009+outer); null=np.array([corr(y,dna+gamma*rng.permutation(rt)) for _ in range(1000)]); dp=corr(y,dna); fp=corr(y,fused)
 rep={'outer':outer,'n_train_oof':len(oof),'n_test':len(test),'pca_components':q,'ridge_alpha':alpha,'gamma_inner_oof':gamma,'oof_residual_corr':corr(residual,ro),'test_residual_corr':corr(y-dna,rt),'dna_PCC':dp,'fusion_PCC':fp,'PCC_gain':fp-dp,'dna_RMSE':float(np.sqrt(np.mean((y-dna)**2))),'fusion_RMSE':float(np.sqrt(np.mean((y-fused)**2))),'permutation_p_one_sided':float((1+np.sum(null>=fp))/(len(null)+1))}; reports.append(rep); print(json.dumps(rep),flush=True)
 z=test[['sample_id','outer','observed','prediction']].rename(columns={'prediction':'dna_prediction'}); z['rna_residual_prediction']=rt; z['gamma']=gamma; z['fusion_prediction']=fused; preds.append(z); nullrows.extend({'outer':outer,'permuted_PCC':float(v)} for v in null)
p=pd.concat(preds,ignore_index=True); assert len(p)==351 and p.sample_id.is_unique; p.to_csv(out/'outer_test_predictions.csv',index=False); pd.DataFrame(reports).to_csv(out/'fold_metrics.csv',index=False); pd.DataFrame(nullrows).to_csv(out/'permutation_distributions.csv',index=False)
summary={'status':'PASS','trait':'A_Freq.Lymph','n':len(p),'folds':5,'mean_fold_dna_PCC':float(np.mean([r['dna_PCC'] for r in reports])),'mean_fold_fusion_PCC':float(np.mean([r['fusion_PCC'] for r in reports])),'mean_fold_gain':float(np.mean([r['PCC_gain'] for r in reports])),'positive_folds':int(sum(r['PCC_gain']>0 for r in reports)),'pooled_dna_PCC':corr(p.observed,p.dna_prediction),'pooled_fusion_PCC':corr(p.observed,p.fusion_prediction),'pooled_gain':corr(p.observed,p.fusion_prediction)-corr(p.observed,p.dna_prediction),'protocol':'5 outer x 3 inner family-grouped; all RNA preprocessing fit on training subset; gamma selected from inner OOF only'}; (out/'STRICT_RNA_FUSION_COMPLETE.json').write_text(json.dumps(summary,indent=2)); print(json.dumps(summary,indent=2))

