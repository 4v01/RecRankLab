"""Generate v3 evidence, task-specific tables and paired user uncertainty."""
import json
import shutil
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from recranklab.data.prepare import digest
from recranklab.evaluation.metrics import paired_bootstrap
from recranklab.utils.repro import provenance

REPORT=Path('reports/v3')
METRICS=['rmse','mae','rmse_clipped_aux','mae_clipped_aux','ndcg@10','recall@10','ndcg@20','recall@20','coverage@10','coverage@20']
RESOURCES=['final_fit_seconds','total_seconds','evaluation_seconds','peak_rss_mib','trainable_parameters']


def main():
    study=yaml.safe_load(Path('configs/v3/study.yaml').read_text())
    REPORT.mkdir(parents=True,exist_ok=True);evidence=REPORT/'evidence';evidence.mkdir(exist_ok=True)
    records=[];run_map={};ranking={};rating={};trials=[];identities=set()
    for file in sorted(Path('outputs/v3/ml-1m').glob('*/*/result.json')):
        r=json.loads(file.read_text());name,seed=r['experiment'],r['seed'];key=f'{name}_seed{seed}'
        if name not in study['experiments'] or key in run_map:raise ValueError('Unplanned/duplicate run')
        if r['environment']['git_dirty']:raise ValueError('Dirty experiment')
        score=r['test_metrics'];identities.add((r['data']['fingerprint'],score['eligible_user_sha256'],score['candidate_catalog_sha256']))
        user_path=file.parent/'per_user.csv'
        if digest(user_path)!=r['per_user_sha256']:raise ValueError('Corrupt ranking rows')
        ranking[name,seed]=pd.read_csv(user_path).set_index('evaluation_row')
        shutil.copyfile(file,evidence/f'{key}.json');shutil.copyfile(user_path,evidence/f'{key}_users.csv')
        if 'per_user_rating_sha256' in r:
            rp=file.parent/'per_user_rating.csv'
            if digest(rp)!=r['per_user_rating_sha256']:raise ValueError('Corrupt rating rows')
            rating[name,seed]=pd.read_csv(rp).set_index('evaluation_row')
            shutil.copyfile(rp,evidence/f'{key}_rating_users.csv')
        run_map[key]={'source_commit':r['environment']['git_commit'],'seed':seed,'config':r['config'],
                      'selected_config':r['selected_config'],'dataset':r['data'],'environment':r['environment'],
                      'evidence':f'evidence/{key}.json'}
        row={k:r[k] for k in ('experiment','model','seed',*RESOURCES)}
        row.update(score);row.update(selection_metric=r['config'].get('selection_metric','ndcg@10'),
                    selected_epoch=r['selected_config']['epochs'] if r['model'] in ('biasmf','autorec','multivae') else None,
                    attempted_trials=len(r['validation_trials']),failed_trials=sum('error' in t for t in r['validation_trials']),
                    validation_checks=sum(len(t.get('checkpoint_history',[])) or int('metrics' in t) for t in r['validation_trials']),
                    source_commit=r['environment']['git_commit'],config_sha256=r['config_sha256'],data_fingerprint=r['data']['fingerprint'])
        records.append(row)
        for t in r['validation_trials']:
            trials.append({'run':key,'trial':t['trial_id'],'selected':t['trial_id']==r['selected_trial'],
                'overrides':json.dumps(t['overrides'],sort_keys=True),'error':t.get('error'),
                'best_epoch':t.get('best_epoch'),'stopped_epoch':t.get('stopped_epoch'),
                'fit_seconds':t['fit_seconds'],**t.get('metrics',{})})
    if not records or len(identities)!=1:raise ValueError('Missing/incomparable evidence')
    frame=pd.DataFrame(records);frame.to_csv(REPORT/'per_seed_results.csv',index=False)
    pd.DataFrame(trials).to_csv(REPORT/'validation_trials.csv',index=False)
    completed={n:sorted(frame.loc[frame.experiment==n,'seed'].tolist()) for n in study['experiments']}
    missing={n:s for n,s in completed.items() if s!=study['seeds']}
    status={'complete':not missing,'completed_seeds':completed,'missing_or_mismatched':missing}
    (REPORT/'completion_status.json').write_text(json.dumps(status,indent=2))
    summaries=[]
    for name,g in frame.groupby('experiment',sort=True):
        row={'experiment':name,'seeds':len(g),'selection_metric':g.selection_metric.iloc[0]}
        for metric in METRICS+RESOURCES+['selected_epoch','validation_checks','failed_trials']:
            values=g[metric].dropna() if metric in g else pd.Series(dtype=float)
            row[metric+'_mean']=values.mean() if len(values) else None
            row[metric+'_std']=values.std(ddof=1) if len(values)>1 else (0. if len(values) else None)
        row['eligible_users']=int(g.eligible_users.iloc[0]);row['rating_count']=int(g.rating_count.iloc[0]) if g.selection_metric.iloc[0]=='rmse' else None
        summaries.append(row)
    summary=pd.DataFrame(summaries);summary.to_csv(REPORT/'results.csv',index=False)
    ratings=summary[summary.selection_metric=='rmse'];rankings=summary[summary.selection_metric=='ndcg@10']
    ratings.to_csv(REPORT/'rating_results.csv',index=False);rankings.to_csv(REPORT/'ranking_results.csv',index=False)
    paired=[]
    for a,b in study['comparisons']:
        if any((a,s) not in ranking or (b,s) not in ranking for s in study['seeds']):continue
        if study['experiments'][a]['objective']=='ndcg@10':
            for metric in ['ndcg@10','recall@10','ndcg@20','recall@20']:
                diffs=[]
                for s in study['seeds']:
                    x,y=ranking[a,s],ranking[b,s]
                    if not x.index.equals(y.index):raise ValueError('Pairing mismatch')
                    diffs.append((x[metric]-y[metric]).to_numpy())
                ci=paired_bootstrap(np.mean(diffs,axis=0))
                paired.append({'comparison':f'{a} - {b}','metric':metric,**ci})
        else:
            sse_a=np.stack([rating[a,s].sse.to_numpy() for s in study['seeds']])
            sse_b=np.stack([rating[b,s].sse.to_numpy() for s in study['seeds']])
            counts=rating[a,study['seeds'][0]]['count'].to_numpy()
            for s in study['seeds']:
                if not np.array_equal(counts,rating[a,s]['count']) or not np.array_equal(counts,rating[b,s]['count']):raise ValueError('Rating population mismatch')
            def delta(idx):return float(np.mean(np.sqrt(sse_a[:,idx].sum(1)/counts[idx].sum())-np.sqrt(sse_b[:,idx].sum(1)/counts[idx].sum())))
            rng=np.random.default_rng(2026);samples=[delta(rng.integers(len(counts),size=len(counts))) for _ in range(2000)]
            lo,hi=np.quantile(samples,[.025,.975])
            paired.append({'comparison':f'{a} - {b}','metric':'rmse','mean_difference':delta(np.arange(len(counts))),
                'ci_low':float(lo),'ci_high':float(hi),'bootstrap_seed':2026,'bootstrap_replicates':2000,'unit':'user; recompute each seed RMSE then average differences'})
    pd.DataFrame(paired).to_csv(REPORT/'paired_differences.csv',index=False)
    figures=REPORT/'figures';figures.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for ax,data,metric,title in [(axes[0],ratings,'rmse','Rating: RMSE-selected; lower is better'),(axes[1],rankings,'ndcg@10','Ranking: NDCG-selected; higher is better')]:
        ax.barh(data.experiment,data[metric+'_mean'],xerr=data[metric+'_std'],capsize=3)
        ax.set(title=title,xlabel=metric)
    fig.tight_layout();fig.savefig(figures/'task_results.png',dpi=160);plt.close(fig)
    old=pd.read_csv('reports/results.csv').set_index('experiment')
    lines=['# V3 actual-results report','','Generated only from new ML-1M run evidence; v2 remains unchanged.',
           f"Frozen v3 study complete: **{not missing}**; final runs: {len(frame)}.",'',
           '## Rating prediction: validation RMSE selection','',
           '| Model | Unclipped RMSE | MAE | Clipped RMSE (auxiliary) | Epoch | Total seconds |',
           '|---|---:|---:|---:|---:|---:|']
    for _,r in ratings.iterrows():
        lines.append(f"| {r.experiment} | {r.rmse_mean:.6f} ± {r.rmse_std:.6f} | {r.mae_mean:.6f} | {r.rmse_clipped_aux_mean:.6f} | {r.selected_epoch_mean:.1f} | {r.total_seconds_mean:.1f} |")
    lines+=['',f"Historical v2 item AutoRec RMSE: {old.loc['autorec_item','rmse_mean']:.6f}; it was NDCG-selected with a different grid and fixed epochs. This descriptive reference does not isolate early stopping or objective selection. V2 rating/model evidence remains in ../evidence/.",
        '', '## Ranking: validation NDCG@10 selection','','| Model | NDCG@10 | Recall@10 | NDCG@20 | Recall@20 | Total seconds |','|---|---:|---:|---:|---:|---:|']
    for _,r in rankings.iterrows():
        lines.append(f"| {r.experiment} | {r['ndcg@10_mean']:.6f} ± {r['ndcg@10_std']:.6f} | {r['recall@10_mean']:.6f} | {r['ndcg@20_mean']:.6f} | {r['recall@20_mean']:.6f} | {r.total_seconds_mean:.1f} |")
    lines+=['','## Paired uncertainty','','| Comparison | Metric | Mean difference | 95% user bootstrap |','|---|---|---:|---:|']
    for p in paired:
        if p['metric'] in ('rmse','ndcg@10'):
            lines.append(f"| {p['comparison']} | {p['metric']} | {p['mean_difference']:.6f} | [{p['ci_low']:.6f}, {p['ci_high']:.6f}] |")
    first=next(iter(run_map.values()));env=first['environment'];pop=frame.iloc[0]
    lines+=['','## Audit and limitations','',
        f"All models share {int(pop.eligible_users)} ranking users, {int(pop.catalog_size)} supported items and {int(pop.cold_start_positive_targets)} cold-positive exclusions. Rating models use {int(ratings.rating_count.iloc[0])} observed test ratings. Data fingerprints and cohort hashes are checked across runs.",
        f"Hardware: {env['cpu']}, {env['logical_cpus']} visible logical CPUs, configured two PyTorch/BLAS threads, CPU only. Python {env['python']}, Torch {env['packages']['torch']}. RSS is process-lifetime high-water memory, not isolated per-model allocation.",
        'Every model has four parameter trials; iterative models additionally search validation checkpoints. Counts and full histories are exported. Compute and validation exposure differ; this is not equal-time or equal-search-effort causal attribution.',
        'SLIM records convergence diagnostics; failures count against budget and are not silently scored. Deterministic classical model seeds are reproducibility repeats. MultiVAE evaluation uses posterior means. Original unobserved feedback, support, masking and Recall/NDCG definitions are unchanged.',
        'Unclipped RMSE remains primary. Clipped metrics are diagnostic only and do not drive selection. External lower RMSE figures use different protocols, and one inspected AutoRec implementation inputs test ratings to its test reconstruction; see research/README.md for exact pinned evidence.',
        'This is a reused benchmark: v1/v2 test results were known. Conditional user bootstrap omits new-data, search and multiple-comparison uncertainty. No v3 test feedback changes the frozen candidates. See methodology.md, experiment_manifest.json and evidence/; data/checkpoints are not redistributed.']
    (REPORT/'technical_report.md').write_text('\n'.join(lines)+'\n')
    smoke=[]
    for f in sorted(Path('outputs/v3_smoke/ml-100k').glob('*/*/result.json')):
        r=json.loads(f.read_text());smoke.append({'experiment':r['experiment'],'model':r['model'],'seed':r['seed'],
            'source_commit':r['environment']['git_commit'],'dataset':r['dataset'],**r['test_metrics']})
    pd.DataFrame(smoke).to_csv(REPORT/'smoke_summary.csv',index=False)
    artifacts={str(p.relative_to(REPORT)):{'sha256':digest(p)} for p in sorted(REPORT.rglob('*'))
               if p.is_file() and p.name!='experiment_manifest.json' and 'research' not in p.parts}
    manifest={'schema_version':3,'generated_at_utc':datetime.now(timezone.utc).isoformat(),'study':study,'status':status,
        'report_generator_environment':provenance(require_clean=False),'runs':run_map,'artifacts':artifacts,
        'historical_reference':{'path':'reports/results.csv','sha256':digest('reports/results.csv'),'manifest_sha256':digest('reports/experiment_manifest.json')}}
    (REPORT/'experiment_manifest.json').write_text(json.dumps(manifest,indent=2))
    print(summary[['experiment','selection_metric','rmse_mean','ndcg@10_mean','selected_epoch_mean']].to_string(index=False))


if __name__=='__main__':main()
