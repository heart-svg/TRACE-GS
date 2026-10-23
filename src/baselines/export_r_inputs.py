#!/usr/bin/env python3
"""Export locked fold-specific S2G features for R GS models."""
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
p=argparse.ArgumentParser(); p.add_argument('--manifest',type=Path,required=True); p.add_argument('--feature-template',required=True); p.add_argument('--out',type=Path,required=True); a=p.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
m=pd.read_csv(a.manifest,dtype={'ID':str})
for o in range(5):
 for s in ('trn','val','tst'):
  q=m[(m.outer==o)&(m.split==s)].copy(); f=pd.read_parquet(Path(a.feature_template.format(outer=o,split=s))); c='ID' if 'ID' in f else f.columns[0]; f[c]=f[c].astype(str); assert not f[c].duplicated().any() and set(f[c])==set(q.ID); f=f.set_index(c).loc[q.ID]; x=f.to_numpy(np.float32); assert np.isfinite(x).all(); z=pd.DataFrame(x,columns=[f'M{i}' for i in range(x.shape[1])]); z.insert(0,'y_raw',q.y_raw.to_numpy()); z.insert(0,'ID',q.ID.to_numpy()); z.to_csv(a.out/f'outer_{o}_{s}.csv.gz',index=False,compression='gzip')
(a.out/'EXPORT_COMPLETE.json').write_text(json.dumps({'status':'PASS','folds':5,'source_manifest':str(a.manifest),'policy':'ID_join_fold_specific_S2G'},indent=2))

