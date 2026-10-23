#!/usr/bin/env python3
"""Strict five-outer-fold RNA residual evaluation for TRACE-GS FT10.

DNA residual targets for the 344 outer-training accessions are constructed only
from strict inner-fold OOF DNA predictions. RNA preprocessing and Ridge tuning
are repeated inside a three-fold CV on those 344 accessions. The outer
validation split selects only the bounded residual scale; test labels are used
once for final reporting and permutation testing.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

ROOT = Path('/path/to/trace_gs_data')
CV = ROOT / 'results/arabidopsis_full/fresh_cv_seed20260902'
STRICT = CV / 'strict_inner_s2g_v1'
TRACE = CV / 'TRACE/FT10'
RNA_PATH = ROOT / 'data/arabidopsis_full/data_prep/omics_transcriptome_log2.parquet'
PHENO_PATH = ROOT / 'data/arabidopsis_full/data_prep/phenotypes.csv'
OUT = CV / 'TRACE/FT10/strict_rna_residual_5fold'
SEED = 20260907
PCA_DIMS = (16, 32, 64)
ALPHAS = (10.0, 100.0, 1000.0, 10000.0)


def corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(pearsonr(a, b).statistic) if np.std(a) and np.std(b) else float('nan')


def fit_project(x_fit, x_other, ncomp, seed):
    scaler = StandardScaler().fit(x_fit)
    a = scaler.transform(x_fit)
    b = scaler.transform(x_other)
    pca = PCA(n_components=min(ncomp, len(x_fit)-1), svd_solver='randomized', random_state=seed)
    return scaler, pca, pca.fit_transform(a), pca.transform(b)


def select_hparams(x, residual, outer):
    cv = KFold(3, shuffle=True, random_state=SEED + outer)
    scores = {(q, a): [] for q in PCA_DIMS for a in ALPHAS}
    for inner, (fit, hold) in enumerate(cv.split(x)):
        for q in PCA_DIMS:
            _, _, zfit, zhold = fit_project(x[fit], x[hold], q, SEED + outer*10 + inner)
            for alpha in ALPHAS:
                pred = Ridge(alpha=alpha).fit(zfit, residual[fit]).predict(zhold)
                scores[(q, alpha)].append(mean_squared_error(residual[hold], pred))
    means = {k: float(np.mean(v)) for k, v in scores.items()}
    best = min(means, key=means.get)
    return best, {f'pca{q}_alpha{a:g}': v for (q, a), v in means.items()}


def load_trace(outer, split):
    p = pd.read_csv(TRACE / f'outer_{outer}/dna/predictions_{split}.csv')
    p['ID'] = p['ID'].astype(str)
    return p[['ID', 'y_raw', 'pred_raw']].rename(columns={'pred_raw': 'dna_pred_raw'})


def main():
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'figures').mkdir(exist_ok=True)
    rna = pd.read_parquet(RNA_PATH)
    id_col = 'ID' if 'ID' in rna.columns else rna.columns[0]
    rna[id_col] = rna[id_col].astype(str)
    assert not rna[id_col].duplicated().any()
    rna = rna.set_index(id_col)
    pheno = pd.read_csv(PHENO_PATH, dtype={'ID': str}).set_index('ID')
    reports, perm_rows = [], []

    for outer in range(5):
        fold_start = time.time()
        oof = pd.read_csv(STRICT / f'outer_{outer}/strict_dna_oof_train.csv')
        oof['ID'] = oof['ID'].astype(str)
        assert len(oof) == 344 and not oof.ID.duplicated().any()
        oof['y_raw'] = pheno.loc[oof.ID, 'FT10'].to_numpy(float)
        oof['residual_raw'] = oof.y_raw - oof.pred_raw
        val, tst = load_trace(outer, 'val'), load_trace(outer, 'tst')
        sets = [set(oof.ID), set(val.ID), set(tst.ID)]
        assert not (sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2])
        assert len(set.union(*sets)) == 645

        xtr = rna.loc[oof.ID].to_numpy(np.float32)
        xval = rna.loc[val.ID].to_numpy(np.float32)
        xtst = rna.loc[tst.ID].to_numpy(np.float32)
        (q, alpha), cv_scores = select_hparams(xtr, oof.residual_raw.to_numpy(), outer)
        scaler = StandardScaler().fit(xtr)
        pca = PCA(n_components=q, svd_solver='randomized', random_state=SEED+outer).fit(scaler.transform(xtr))
        ztr = pca.transform(scaler.transform(xtr))
        model = Ridge(alpha=alpha).fit(ztr, oof.residual_raw)
        rval = model.predict(pca.transform(scaler.transform(xval)))
        rtst = model.predict(pca.transform(scaler.transform(xtst)))

        gammas = np.linspace(0, 1.5, 151)
        gamma = float(min(gammas, key=lambda g: mean_squared_error(val.y_raw, val.dna_pred_raw + g*rval)))
        fused = tst.dna_pred_raw.to_numpy() + gamma*rtst
        y = tst.y_raw.to_numpy()
        dna = tst.dna_pred_raw.to_numpy()
        rng = np.random.default_rng(SEED + outer)
        perm = np.array([corr(y, dna + gamma*rng.permutation(rtst)) for _ in range(1000)])
        observed = corr(y, fused)
        report = {
            'outer': outer, 'n_train': len(oof), 'n_val': len(val), 'n_test': len(tst),
            'pca_components': q, 'ridge_alpha': alpha, 'gamma_validation': gamma,
            'train_strict_oof_residual_corr': corr(oof.residual_raw, model.predict(ztr)),
            'val_residual_corr': corr(val.y_raw-val.dna_pred_raw, rval),
            'test_residual_corr': corr(y-dna, rtst),
            'dna_PCC': corr(y, dna), 'fusion_PCC': observed,
            'PCC_gain': observed-corr(y, dna),
            'dna_RMSE': float(np.sqrt(np.mean((y-dna)**2))),
            'fusion_RMSE': float(np.sqrt(np.mean((y-fused)**2))),
            'dna_MAE': float(np.mean(np.abs(y-dna))),
            'fusion_MAE': float(np.mean(np.abs(y-fused))),
            'permuted_PCC_mean': float(perm.mean()),
            'permuted_PCC_sd': float(perm.std()),
            'permutation_p_one_sided': float((1+np.sum(perm>=observed))/(len(perm)+1)),
            'seconds': time.time()-fold_start,
            'inner_cv_mse': cv_scores,
        }
        reports.append(report)
        pred = pd.DataFrame({'ID':tst.ID, 'true_raw':y, 'dna_pred_raw':dna,
                             'rna_residual_pred_raw':rtst, 'fusion_pred_raw':fused})
        pred.to_csv(OUT/f'outer_{outer}_predictions.csv', index=False)
        (OUT/f'outer_{outer}_metrics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        perm_rows.extend({'outer':outer, 'permuted_PCC':v} for v in perm)
        print('OUTER_PASS', json.dumps({k:report[k] for k in ['outer','pca_components','ridge_alpha','gamma_validation','dna_PCC','fusion_PCC','PCC_gain','permutation_p_one_sided']}), flush=True)

    summary = pd.DataFrame([{k:v for k,v in r.items() if k != 'inner_cv_mse'} for r in reports])
    summary.to_csv(OUT/'fivefold_metrics.csv', index=False)
    pd.DataFrame(perm_rows).to_csv(OUT/'permutation_distributions.csv', index=False)
    aggregate = {'status':'PASS', 'elapsed_seconds':time.time()-start, 'folds':5}
    for key in ['dna_PCC','fusion_PCC','PCC_gain','dna_RMSE','fusion_RMSE','dna_MAE','fusion_MAE']:
        aggregate[key]={'mean':float(summary[key].mean()), 'sd':float(summary[key].std(ddof=1))}
    aggregate['positive_gain_folds'] = int((summary.PCC_gain>0).sum())
    (OUT/'aggregate_metrics.json').write_text(json.dumps(aggregate, indent=2), encoding='utf-8')

    plt.rcParams.update({'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False})
    colors=['#4472C4','#70AD47']
    x=np.arange(5)
    fig,ax=plt.subplots(figsize=(7.2,4.4))
    ax.plot(x,summary.dna_PCC,'o-',lw=2,color=colors[0],label=f"DNA (mean={summary.dna_PCC.mean():.3f})")
    ax.plot(x,summary.fusion_PCC,'o-',lw=2,color=colors[1],label=f"DNA+RNA (mean={summary.fusion_PCC.mean():.3f})")
    for i in x: ax.plot([i,i],[summary.dna_PCC[i],summary.fusion_PCC[i]],color='#BBBBBB',lw=.8)
    ax.set(xticks=x,xlabel='Outer fold',ylabel='PCC',title='TRACE-GS strict nested five-fold performance'); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(OUT/'figures/strict_fivefold_pcc.png',dpi=300); plt.close(fig)

    fig,ax=plt.subplots(1,2,figsize=(9.5,4))
    ax[0].bar(x,summary.PCC_gain,color=np.where(summary.PCC_gain>=0,'#70AD47','#C00000')); ax[0].axhline(0,color='black',lw=.8)
    ax[0].set(xticks=x,xlabel='Outer fold',ylabel='PCC gain',title='RNA incremental gain')
    width=.35
    ax[1].bar(x-width/2,summary.dna_RMSE,width,label='DNA',color=colors[0]); ax[1].bar(x+width/2,summary.fusion_RMSE,width,label='DNA+RNA',color=colors[1])
    ax[1].set(xticks=x,xlabel='Outer fold',ylabel='RMSE (days)',title='Prediction error'); ax[1].legend(frameon=False)
    fig.tight_layout(); fig.savefig(OUT/'figures/strict_gain_and_rmse.png',dpi=300); plt.close(fig)

    perm_df=pd.DataFrame(perm_rows)
    fig,ax=plt.subplots(figsize=(7.2,4.4))
    for outer in range(5):
        vals=perm_df.loc[perm_df.outer==outer,'permuted_PCC']; ax.hist(vals,bins=25,alpha=.25,label=f'Outer {outer}')
        ax.axvline(summary.loc[outer,'fusion_PCC'],lw=1.2)
    ax.set(xlabel='PCC after RNA permutation',ylabel='Frequency',title='RNA sample-permutation test'); ax.legend(frameon=False,ncol=2)
    fig.tight_layout(); fig.savefig(OUT/'figures/strict_rna_permutation.png',dpi=300); plt.close(fig)
    print('STRICT_TRACE_RNA_5FOLD_PASS', json.dumps(aggregate), flush=True)


if __name__ == '__main__':
    main()

