#!/usr/bin/env python3
"""Build an exact ID-level five-fold manifest from formal TRACE predictions."""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--trace-root',required=True,type=Path)
    ap.add_argument('--out',required=True,type=Path)
    ap.add_argument('--species',required=True)
    ap.add_argument('--trait',required=True)
    a=ap.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
    rows=[]; labels={}; test_seen=[]
    for outer in range(5):
        sets={}
        for split in ('trn','val','tst'):
            p=a.trace_root/f'outer_{outer}/dna/predictions_{split}.csv'
            if not p.is_file(): raise FileNotFoundError(p)
            d=pd.read_csv(p,dtype={'ID':str})
            if not {'ID','y_raw'} <= set(d.columns): raise ValueError(f'missing ID/y_raw: {p}')
            if d.ID.duplicated().any(): raise AssertionError(f'duplicate IDs: {p}')
            if not np.isfinite(d.y_raw.to_numpy(float)).all(): raise AssertionError(f'nonfinite labels: {p}')
            sets[split]=set(d.ID)
            for sid,y in zip(d.ID,d.y_raw):
                if sid in labels and abs(labels[sid]-float(y))>1e-5: raise AssertionError(f'label mismatch: {sid}')
                labels[sid]=float(y); rows.append({'ID':sid,'y_raw':float(y),'outer':outer,'split':split})
            if split=='tst': test_seen.extend(d.ID.tolist())
        if sets['trn']&sets['val'] or sets['trn']&sets['tst'] or sets['val']&sets['tst']:
            raise AssertionError(f'overlap in outer {outer}')
        if set.union(*sets.values()) != set(labels):
            # At outer 0 labels is complete; later folds must cover the same cohort.
            if outer>0: raise AssertionError(f'cohort mismatch in outer {outer}')
    if len(test_seen)!=len(set(test_seen)) or set(test_seen)!=set(labels):
        raise AssertionError('each material must occur in exactly one outer test fold')
    tab=pd.DataFrame(rows).sort_values(['outer','split','ID'])
    tab.to_csv(a.out/'split_manifest_long.csv',index=False)
    fold=pd.DataFrame({'ID':list(labels),'y_raw':[labels[x] for x in labels]})
    fold['outer_test']=fold.ID.map({sid:o for o in range(5) for sid in tab[(tab.outer==o)&(tab.split=='tst')].ID})
    fold.sort_values('ID').to_csv(a.out/'samples_and_outer_fold.csv',index=False)
    audit={'status':'PASS','species':a.species,'trait':a.trait,'n_unique':len(labels),'folds':5,
           'source_trace_root':str(a.trace_root),'test_once':True,'id_join_only':True}
    (a.out/'MANIFEST_LOCKED.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print(json.dumps(audit,indent=2))
if __name__=='__main__': main()


