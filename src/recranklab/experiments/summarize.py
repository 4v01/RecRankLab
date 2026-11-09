"""Regenerate v2 tables, uncertainty, figures and audit manifest from actual runs."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from recranklab.data.prepare import digest
from recranklab.evaluation.metrics import paired_bootstrap
from recranklab.utils.repro import provenance

METRICS = ['recall@10','ndcg@10','recall@20','ndcg@20','coverage@10','coverage@20','rmse','mae']
RESOURCES = ['trainable_parameters','final_fit_seconds','evaluation_seconds','total_seconds','peak_rss_mib']


def summarize(root, output):
    report=Path(output).parent; report.mkdir(parents=True,exist_ok=True)
    evidence=report/'evidence'; evidence.mkdir(exist_ok=True)
    figures=report/'figures'; figures.mkdir(exist_ok=True)
    study=yaml.safe_load(Path('configs/study.yaml').read_text())
    files=sorted(Path(root).glob('ml-1m/*/*/result.json'))
    if not files: raise ValueError('No completed new ML-1M runs')
    rows=[]; users={}; runs={}; originals={}
    for file in files:
        run=json.loads(file.read_text()); name=run['experiment']; seed=run['seed']
        if name not in study['experiments']: raise ValueError(f'Unplanned run: {name}')
        if run['environment']['git_dirty']: raise ValueError(f'Dirty source: {file}')
        key=f'{name}_seed{seed}'
        if key in runs: raise ValueError(f'Duplicate: {key}')
        up=file.parent/'per_user.csv'
        if digest(up)!=run['per_user_sha256']: raise ValueError(f'Corrupt metrics: {up}')
        per=pd.read_csv(up).set_index('evaluation_row')
        users[name,seed]=per
        shutil.copyfile(file,evidence/f'{key}.json');shutil.copyfile(up,evidence/f'{key}_users.csv')
        entry={k:run[k] for k in ('model','seed','config','selected_config','config_sha256','completed_at_utc','data','environment')}
        entry['evidence']=f'evidence/{key}.json'; runs[key]=entry; originals[key]=run
        row={k:run[k] for k in ('dataset','experiment','model','seed','selected_trial',*RESOURCES)}
        row.update(run['test_metrics'])
        row.update(git_commit=run['environment']['git_commit'],data_fingerprint=run['data']['fingerprint'],
                   config_sha256=run['config_sha256'],validation_fits=len(run['validation_trials']),
                   epochs=run['selected_config']['epochs'] if run['model'] not in ('popularity','ease') else None)
        rows.append(row)
    frame=pd.DataFrame(rows)
    for field in ['data_fingerprint','eligible_user_sha256','candidate_catalog_sha256','eligible_users']:
        if frame[field].nunique()!=1: raise ValueError(f'Incomparable runs: {field}')
    frame.to_csv(report/'per_seed_results.csv',index=False)
    completed={name:sorted(frame.loc[frame.experiment==name,'seed'].tolist()) for name in study['experiments']}
    missing={name:seeds for name,seeds in completed.items() if seeds!=study['seeds']}
    status={'complete':not missing,'expected_seeds':study['seeds'],'completed_seeds':completed,'missing_or_mismatched':missing}
    (report/'completion_status.json').write_text(json.dumps(status,indent=2))
    aggregate=[]
    for name,group in frame.groupby('experiment',sort=True):
        row={'experiment':name,'seeds':len(group),'validation_fits_per_seed':int(group.validation_fits.iloc[0])}
        for metric in METRICS+RESOURCES:
            vals=group[metric].dropna() if metric in group else pd.Series(dtype=float)
            row[metric+'_mean']=float(vals.mean()) if len(vals) else None
            row[metric+'_std']=float(vals.std(ddof=1)) if len(vals)>1 else (0. if len(vals) else None)
        for field in ['eligible_users','cold_start_positive_targets','catalog_size','heldout_positive_targets','unsupported_catalog_items']:
            row[field]=int(group[field].iloc[0])
        aggregate.append(row)
    summary=pd.DataFrame(aggregate);summary.to_csv(output,index=False)
    paired=[]
    for a,b in study['comparisons']:
        seeds=sorted(set(completed[a])&set(completed[b]))
        for metric in ['recall@10','ndcg@10','recall@20','ndcg@20']:
            diffs=[]
            for seed in seeds:
                x,y=users[a,seed],users[b,seed]
                if not x.index.equals(y.index): raise ValueError('Unpaired users')
                diffs.append((x[metric]-y[metric]).to_numpy())
            if diffs:
                paired.append({'comparison':a+' - '+b,'metric':metric,'seeds':len(seeds),
                               **paired_bootstrap(np.mean(diffs,axis=0))})
    pd.DataFrame(paired).to_csv(report/'paired_differences.csv',index=False)
    summary[summary.experiment.isin(['bpr_positives_only','ease','ease_selfopt','autorec_item'])].to_csv(report/'ablations.csv',index=False)
    validation=[]
    for key,run in originals.items():
        for trial in run['validation_trials']:
            validation.append({'run':key,'trial':trial['trial_id'],'selected':trial['trial_id']==run['selected_trial'],
                               'overrides':json.dumps(trial['overrides'],sort_keys=True),
                               'hypothesis':json.dumps(trial.get('hypothesis',{}),sort_keys=True),
                               'fit_seconds':trial['fit_seconds'],**trial['metrics']})
    pd.DataFrame(validation).to_csv(report/'validation_trials.csv',index=False)
    fig,ax=plt.subplots(figsize=(10,6))
    data=summary.sort_values('ndcg@10_mean')
    ax.barh(data.experiment,data['ndcg@10_mean'],xerr=data['ndcg@10_std'],capsize=3)
    ax.set(xlabel='Full-catalog test NDCG@10 (mean ± seed SD)',title='MovieLens 1M • budgets differ; see table')
    fig.tight_layout();fig.savefig(figures/'ranking_comparison.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4))
    for name in ['biasmf','autorec','autorec_item']:
        sub=frame[frame.experiment==name]
        ax.scatter(sub.rmse,sub['ndcg@10'],label=name)
    ax.set(xlabel='Observed-rating RMSE',ylabel='Full-catalog NDCG@10');ax.legend()
    fig.tight_layout();fig.savefig(figures/'rating_vs_ranking.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4))
    for key,run in originals.items():
        if run['experiment'] in ['ease','ease_selfopt'] and run['seed']==17:
            values=[t['metrics']['ndcg@10'] for t in run['validation_trials']]
            ax.plot(range(1,len(values)+1),np.maximum.accumulate(values),marker='o',label=run['experiment'])
    ax.set(xlabel='Validation fits spent',ylabel='Incumbent validation NDCG@10',xticks=[1,2,3,4]);ax.legend()
    fig.tight_layout();fig.savefig(figures/'search_budget.png',dpi=160);plt.close(fig)
    lines=['# RecRankLab v2 — actual experiment report','',
           f'Completed {len(frame)} new ML-1M final fits. Frozen study complete: **{not missing}**.',
           '', 'Historical evidence in legacy_v1 is excluded. Test scores from v1 were known before designing v2; this is a reused benchmark, not an untouched confirmation set.',
           '', '| Model | Seeds | Validation fits/seed | Recall@10 | NDCG@10 | Recall@20 | NDCG@20 | RMSE | Final fit s | Total s |',
           '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for _,r in summary.iterrows():
        cells=[]
        for m in ['recall@10','ndcg@10','recall@20','ndcg@20','rmse','final_fit_seconds','total_seconds']:
            v=r[m+'_mean'];cells.append('—' if pd.isna(v) else f"{v:.6f} ± {r[m+'_std']:.6f}")
        lines.append(f"| {r.experiment} | {r.seeds} | {r.validation_fits_per_seed} | "+' | '.join(cells)+' |')
    audit=frame.iloc[0]
    lines+=['',f"Every model shares {int(audit.eligible_users)} eligible users, {int(audit.catalog_size)} supported items, and {int(audit.cold_start_positive_targets)} excluded cold positives out of {int(audit.heldout_positive_targets)} held-out positives. Full catalog; prior train+validation interactions masked.",
            '', '## Paired user bootstrap', '',
            'Seed-averaged user differences; 2,000 user resamples, seed 2026; conditional percentile intervals. Not corrected for benchmark reuse, multiple comparisons or search uncertainty.',
            '', '| Comparison | NDCG@10 difference | 95% interval |','|---|---:|---:|']
    for p in paired:
        if p['metric']=='ndcg@10': lines.append(f"| {p['comparison']} | {p['mean_difference']:.6f} | [{p['ci_low']:.6f}, {p['ci_high']:.6f}] |")
    first=next(iter(originals.values()))
    env=first['environment']
    item_count=first['data']['n_items']
    lines+=['', '## Hardware and software', '',
            f"Runs used {env['cpu']}; {env['logical_cpus']} visible logical CPUs, two configured PyTorch/BLAS threads, CPU only. The container was observed with a two-CPU quota and 8 GiB memory limit. Python {env['python']}, PyTorch {env['packages']['torch']}, NumPy {env['packages']['numpy']}, pandas {env['packages']['pandas']}, SciPy {env['packages']['scipy']}. These are shared-cloud wall times, not isolated hardware performance measurements.",
            '', f'EASE stores a {item_count}×{item_count} float32 coefficient matrix ({item_count**2*4/1024**2:.2f} MiB; {item_count*(item_count-1):,} off-diagonal entries), in addition to inputs and cached scores. Its zero gradient-parameter count does not mean zero learned storage.',
            '', '## Interpretation and limits','',
            'Original-budget core baselines have two validation trials and 20 epochs. Improvement variants have four trials and longer training; their differences combine architecture, optimizer and budget. Do not describe them as equal-compute causal comparisons. EASE has four analytical fits; its learned dense matrix is not counted as gradient-trained parameters and scales quadratically in item count.',
            '', 'The negative-policy contrast and equal-four-fit EASE grid/memory-search comparison are reported in ablations.csv. The self-optimizer uses validation hypotheses and elite memory; a lower or tied test result remains in the report. It is not a CogAlpha/XALPHA reproduction or LLM system. Deterministic EASE seeds are duplicate checks, not independent uncertainty draws.',
            '', 'The explicit RMSE and ranking scatter compares different targets/populations and is descriptive. No universal relation or universal superiority follows. All unsupported targets and users follow the shared evaluator.',
            '', 'See methodology.md for frozen rules, limitations.md for known issues, validation_trials.csv for all successful/failed search candidates, per_seed_results.csv for hardware/resource metrics, and experiment_manifest.json/evidence for commit/config/data/seed provenance. Checkpoints, splits and raw data are not distributed. RSS includes earlier trials in the process; final fit excludes tuning, total time includes it. Optional LLM reranking remains deferred.']
    (report/'technical_report.md').write_text('\n'.join(lines)+'\n')
    smoke=[]
    for file in sorted(Path('outputs/smoke/ml-100k').glob('*/*/result.json')):
        s=json.loads(file.read_text());key=f"smoke_{s['model']}_seed{s['seed']}"
        smoke.append({'model':s['model'],'seed':s['seed'],'epochs':s['selected_config']['epochs'],**s['test_metrics']})
        shutil.copyfile(file,evidence/f'{key}.json')
        runs[key]={k:s[k] for k in ('model','seed','config','selected_config','config_sha256','completed_at_utc','data','environment')}
        runs[key]['evidence']=f'evidence/{key}.json'
    pd.DataFrame(smoke).to_csv(report/'smoke_summary.csv',index=False)
    artifacts={}
    for path in sorted(report.rglob('*')):
        if not path.is_file() or path.name=='experiment_manifest.json' or any(p in path.parts for p in ('legacy_v1','external_benchmark','v3')): continue
        key=path.stem.removesuffix('_users')
        artifacts[str(path.relative_to(report))]={'sha256':digest(path),'source_runs':[key] if key in runs else list(originals)}
    manifest={'schema_version':2,'generated_at_utc':datetime.now(timezone.utc).isoformat(),
              'report_generator_environment':provenance(require_clean=False),'study':study,'status':status,
              'runs':runs,'artifacts':artifacts}
    (report/'experiment_manifest.json').write_text(json.dumps(manifest,indent=2))
    print(summary[['experiment','seeds','ndcg@10_mean','ndcg@10_std','rmse_mean']].to_string(index=False))


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default='outputs/main');p.add_argument('--output',default='reports/results.csv')
    a=p.parse_args();summarize(a.root,a.output)


if __name__=='__main__':main()
