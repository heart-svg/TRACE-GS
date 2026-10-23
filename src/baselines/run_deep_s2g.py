#!/usr/bin/env python3
"""Leakage-controlled PyTorch adapters for DeepGS, LCNN and Cropformer."""
import argparse,json,random,time
from pathlib import Path
import numpy as np,pandas as pd,torch
from scipy.stats import pearsonr
from torch import nn
from torch.utils.data import DataLoader,TensorDataset

def pcc(y,p):
 y,p=np.asarray(y,float),np.asarray(p,float)
 if not np.isfinite(y).all() or not np.isfinite(p).all() or np.std(y)<1e-12 or np.std(p)<1e-8:return 0.
 r=float(pearsonr(y,p).statistic)
 return r if np.isfinite(r) else 0.
def met(y,p):
 y,p=np.asarray(y,float),np.asarray(p,float); return {'PCC':pcc(y,p),'RMSE':float(np.sqrt(np.mean((y-p)**2))),'MAE':float(np.mean(abs(y-p)))}
def load(path,ids):
 d=pd.read_parquet(path); c='ID' if 'ID' in d else d.columns[0]; d[c]=d[c].astype(str); assert not d[c].duplicated().any(); d=d.set_index(c); assert set(d.index)==set(ids); x=d.loc[ids].to_numpy(np.float32); assert np.isfinite(x).all(); return x
class DeepGS(nn.Module):
 def __init__(self,p):
  super().__init__(); self.f=nn.Sequential(nn.Conv1d(1,8,18),nn.ReLU(),nn.MaxPool1d(4),nn.Dropout(.2)); n=(p-17)//4; self.h=nn.Sequential(nn.Flatten(),nn.Linear(8*n,32),nn.Sigmoid(),nn.Dropout(.1),nn.Linear(32,1))
 def forward(self,x): return self.h(self.f(x[:,None])).squeeze(1)
class LocalLayer(nn.Module):
 def __init__(self,p,k=10,c=4): super().__init__(); self.k=k; self.w=nn.Parameter(torch.randn(p-k+1,k,c)*.02); self.b=nn.Parameter(torch.zeros(p-k+1,c))
 def forward(self,x): return torch.einsum('blk,lkc->blc',x.unfold(1,self.k,1),self.w)+self.b
class LCNN(nn.Module):
 def __init__(self,p): super().__init__(); self.local=LocalLayer(p); n=(p-9)//4; self.h=nn.Sequential(nn.ReLU(),nn.MaxPool1d(4),nn.Flatten(),nn.Linear(4*n,64),nn.ReLU(),nn.Dropout(.2),nn.Linear(64,1))
 def forward(self,x): return self.h(self.local(x).transpose(1,2)).squeeze(1)
class Cropformer(nn.Module):
 def __init__(self,p,tok=128,d=64):
  super().__init__(); self.tok=tok; self.pad=(-p)%tok; self.emb=nn.Linear(4,d); self.conv=nn.Conv1d(d,d,3,padding=1); self.att=nn.MultiheadAttention(d,4,batch_first=True); self.norm=nn.LayerNorm(d); self.head=nn.Sequential(nn.Linear(d,64),nn.ReLU(),nn.Dropout(.2),nn.Linear(64,1))
 def forward(self,x):
  if self.pad: x=nn.functional.pad(x,(0,self.pad))
  z=x.reshape(len(x),self.tok,-1)
  z=torch.stack((z.mean(2),z.std(2,unbiased=False),z.amin(2),z.amax(2)),2)
  z=self.emb(z); z=self.conv(z.transpose(1,2)).transpose(1,2)
  a,_=self.att(z,z,z)
  return self.head(self.norm(z+a).mean(1)).squeeze(1)
def main():
 a=argparse.ArgumentParser()
 a.add_argument('--manifest',type=Path,required=True); a.add_argument('--feature-template',required=True)
 a.add_argument('--model',choices=['deepgs','lcnn','cropformer'],required=True); a.add_argument('--out',type=Path,required=True)
 a.add_argument('--epochs',type=int,default=200); a.add_argument('--patience',type=int,default=25); a.add_argument('--seed',type=int,default=20260902)
 z=a.parse_args(); z.out.mkdir(parents=True,exist_ok=True)
 man=pd.read_csv(z.manifest,dtype={'ID':str}); rows=[]; pooled=[]; start=time.time(); dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 configs=[(1e-3,1e-4),(3e-4,1e-4),(1e-4,1e-4),(1e-3,1e-3),(3e-4,1e-3)]
 cls={'deepgs':DeepGS,'lcnn':LCNN,'cropformer':Cropformer}[z.model]
 for outer in range(5):
  parts={s:man[(man.outer==outer)&(man.split==s)].copy() for s in ('trn','val','tst')}
  assert all(len(parts[s]) for s in parts)
  assert not set(parts['trn'].ID)&set(parts['val'].ID) and not set(parts['trn'].ID)&set(parts['tst'].ID) and not set(parts['val'].ID)&set(parts['tst'].ID)
  X={s:load(Path(z.feature_template.format(outer=outer,split=s)),parts[s].ID.tolist()) for s in parts}
  mu=X['trn'].mean(0); sd=X['trn'].std(0); sd[sd<1e-6]=1; X={s:(v-mu)/sd for s,v in X.items()}
  ym=float(parts['trn'].y_raw.mean()); ys=float(parts['trn'].y_raw.std() or 1.)
  ytr=torch.tensor(((parts['trn'].y_raw-ym)/ys).to_numpy(np.float32)); best_trial=None; trial_reports=[]
  for trial,(lr,wd) in enumerate(configs):
   trial_seed=z.seed+outer*100+trial; random.seed(trial_seed); np.random.seed(trial_seed); torch.manual_seed(trial_seed)
   if torch.cuda.is_available():torch.cuda.manual_seed_all(trial_seed)
   m=cls(X['trn'].shape[1]).to(dev); opt=torch.optim.AdamW(m.parameters(),lr=lr,weight_decay=wd); loss=nn.MSELoss()
   gen=torch.Generator().manual_seed(trial_seed); dl=DataLoader(TensorDataset(torch.tensor(X['trn']),ytr),batch_size=32,shuffle=True,generator=gen)
   best=1e99; left=z.patience; state=None; hist=[]; be=0
   for ep in range(1,z.epochs+1):
    m.train(); ls=[]
    for xb,yb in dl:
     opt.zero_grad(); l=loss(m(xb.to(dev)),yb.to(dev)); l.backward(); nn.utils.clip_grad_norm_(m.parameters(),1); opt.step(); ls.append(float(l.detach().cpu()))
    m.eval()
    with torch.no_grad():pv=m(torch.tensor(X['val']).to(dev)).cpu().numpy()*ys+ym
    vm=float(np.mean((parts['val'].y_raw.to_numpy()-pv)**2)); hist.append({'trial':trial,'epoch':ep,'train_loss':float(np.mean(ls)),'val_mse':vm})
    if vm<best-1e-8:best=vm;left=z.patience;state={k:v.detach().cpu().clone() for k,v in m.state_dict().items()};be=ep
    else:left-=1
    if left<=0:break
   report={'trial':trial,'lr':lr,'weight_decay':wd,'best_epoch':be,'val_mse':best,'history':hist,'state':state}
   trial_reports.append(report)
   if best_trial is None or best<best_trial['val_mse']:best_trial=report
   del m
  m=cls(X['trn'].shape[1]).to(dev); m.load_state_dict(best_trial['state']); m.eval()
  with torch.no_grad():pt=m(torch.tensor(X['tst']).to(dev)).cpu().numpy()*ys+ym
  mm=met(parts['tst'].y_raw,pt); rep={'outer':outer,'model':z.model,'selected_trial':best_trial['trial'],'lr':best_trial['lr'],'weight_decay':best_trial['weight_decay'],'best_epoch':best_trial['best_epoch'],'val_mse':best_trial['val_mse'],'n_train':len(parts['trn']),'n_val':len(parts['val']),'n_test':len(parts['tst']),**mm}; rows.append(rep)
  od=z.out/f'outer_{outer}'; od.mkdir(exist_ok=True)
  pd.concat([pd.DataFrame(r['history']) for r in trial_reports],ignore_index=True).to_csv(od/'all_trial_history.csv',index=False)
  pd.DataFrame([{k:v for k,v in r.items() if k not in ('history','state')} for r in trial_reports]).to_csv(od/'trial_summary.csv',index=False)
  pp=pd.DataFrame({'ID':parts['tst'].ID,'outer':outer,'y_raw':parts['tst'].y_raw,'pred_raw':pt}); pp.to_csv(od/'predictions_tst.csv',index=False); pooled.append(pp); (od/'metrics.json').write_text(json.dumps(rep,indent=2)); print(rep,flush=True)
 tab=pd.DataFrame(rows); pred=pd.concat(pooled,ignore_index=True); assert not pred.ID.duplicated().any()
 agg={'status':'PASS','model':z.model,'implementation':'audited_PyTorch_S2G_adapter','hpo_trials_per_fold':5,'selection':'validation_MSE_only','folds':5,'PCC_mean':float(tab.PCC.mean()),'PCC_sd':float(tab.PCC.std(ddof=1)),'RMSE_mean':float(tab.RMSE.mean()),'MAE_mean':float(tab.MAE.mean()),'pooled':met(pred.y_raw,pred.pred_raw),'elapsed_seconds':time.time()-start}
 tab.to_csv(z.out/'fivefold_metrics.csv',index=False); pred.to_csv(z.out/'oof_predictions.csv',index=False); (z.out/'COMPLETE.json').write_text(json.dumps(agg,indent=2)); print(agg)
if __name__=='__main__':main()

