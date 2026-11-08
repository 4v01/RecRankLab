"""Generate extended diagnostics exclusively from saved run evidence."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd


def generate(report="reports"):
    report=Path(report)
    results=pd.read_csv(report/'results.csv').set_index('experiment')
    runs=[json.loads(p.read_text()) for p in sorted((report/'evidence').glob('*.json')) if not p.name.startswith('smoke_')]
    rows=[]
    fig,axes=plt.subplots(2,2,figsize=(10,7))
    for ax,name in zip(axes.flat,['biasmf','autorec','bpr_mf','lightgcn']):
        for run in runs:
            if run['experiment']==name:
                values=run['final_loss_history']
                ax.plot(range(1,len(values)+1),values,label=str(run['seed']))
        ax.set(title=name,xlabel='Epoch',ylabel='Model-specific training objective');ax.legend(title='Seed')
    fig.suptitle('Fixed 20-epoch refits — loss scales are not comparable across models')
    fig.tight_layout();fig.savefig(report/'figures/training_curves.png',dpi=160);plt.close(fig)
    for run in runs:
        for trial in run['validation_trials']:
            rows.append({'experiment':run['experiment'],'seed':run['seed'],'trial':trial['trial_id'],
                         'selected':trial['trial_id']==run['selected_trial'],
                         'lr':trial['overrides'].get('lr'), 'validation_ndcg@10':trial['metrics']['ndcg@10'],
                         'validation_rmse':trial['metrics'].get('rmse'), 'fit_seconds':trial['fit_seconds']})
    pd.DataFrame(rows).to_csv(report/'validation_trials.csv',index=False)
    b,a,l=results.loc['bpr_mf'],results.loc['autorec'],results.loc['lightgcn']
    pop=results.loc['popularity','ndcg@10_mean']
    text=['# Findings and diagnostic interpretation','',
          'These observations were written after the predeclared runs. They did not trigger model, split, candidate, epoch or test changes. Numeric values below are generated from saved evidence.', '',
          f"BPR-MF mean NDCG@10 is {b['ndcg@10_mean']:.6f}; Popularity is {pop:.6f}; their absolute difference is {b['ndcg@10_mean']-pop:.6f}. This is an implementation-specific fixed-budget observation, not a claim of general superiority.", '',
          f"AutoRec mean RMSE is {a['rmse_mean']:.6f} and NDCG@10 is {a['ndcg@10_mean']:.6f} with seed SD {a['ndcg@10_std']:.6f}. Selection differs across seeds; see validation_trials.csv. The model's sensitivity to initialization and the small search budget limits stable conclusions.", '',
          f"LightGCN minus BPR mean NDCG@10 is {l['ndcg@10_mean']-b['ndcg@10_mean']:.6f}. LightGCN final fit takes {l['final_fit_seconds_mean']:.2f}s on average versus {b['final_fit_seconds_mean']:.2f}s for BPR. See paired_differences.csv for the user-level interval, not only the ranking of means.", '',
          '## Rating error is not the selection objective', '',
          'For AutoRec seed 17, the two validation trials provide a concrete objective-mismatch example:']
    run=next(r for r in runs if r['experiment']=='autorec' and r['seed']==17)
    for trial in run['validation_trials']:
        text.append(f"- Learning rate {trial['overrides']['lr']}: validation RMSE {trial['metrics']['rmse']:.6f}; NDCG@10 {trial['metrics']['ndcg@10']:.6f}.")
    text += ['', 'The chosen trial maximizes validation NDCG@10 even if the other trial has a lower RMSE. This is a local counterexample to equating the two metrics, not an estimate of a universal relationship.', '',
             '## Observed weaknesses', '',
             'BiasMF and AutoRec both rank below Popularity under the tested setup. The observed-rating objective does not penalize predictions on missing items. BiasMF’s final training objective is substantially lower than held-out squared error; overfitting and temporal distribution shift are plausible contributors. This does not establish which mechanism dominates. No additional test-guided tuning was performed.', '',
             'Training curves are included for all four trainable headline systems. Declining training loss alone does not establish converged validation ranking. These model-specific objectives have different scales and should not be compared numerically with one another.', '',
             '## Follow-up experiments, not completed claims', '',
             'A future preregistered study could broaden regularization and epoch searches using validation only, compare U-AutoRec with I-AutoRec, and add a globally forward split. These follow-ups must be reported separately; the existing test set should not become an informal tuning set. Optional LLM reranking remains deferred.']
    (report/'findings.md').write_text('\n'.join(text)+'\n')


if __name__=='__main__':
    generate()
