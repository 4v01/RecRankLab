"""Descriptive diagnostics; no model or tuning decisions use new test labels."""
import gc,json
from pathlib import Path
import numpy as np
from recranklab.data.prepare import load
from recranklab.models.scalable import fit_biases

out={'scope':'data structure and train-to-validation diagnostic; historical v3 final results already known','datasets':{},'v3_stopping':{}}
for dataset in ('ml-1m','ml-10m'):
    parts,m=load(dataset);train,val=parts['train'],parts['val']
    bu,bi=fit_biases(train,m['n_users'],m['n_items'])
    truth=val.rating.to_numpy();prediction=train.rating.mean()+bu[val.user]+bi[val.item]
    sse=np.square(prediction-truth).sum();sst=np.square(truth-truth.mean()).sum()
    counts=train.groupby('user').size().to_numpy()
    pos=val[val.rating>=4].groupby('user').size().to_numpy()
    item_counts=train.item.value_counts();cold=~val.item.isin(item_counts.index)
    cohorts=[]
    for name,mask in [('cold',cold),('1_to_19_fit_ratings',val.item.map(item_counts).fillna(0).between(1,19)),('20_to_99_fit_ratings',val.item.map(item_counts).fillna(0).between(20,99)),('100plus_fit_ratings',val.item.map(item_counts).fillna(0)>=100)]:
        mask=mask.to_numpy();cohorts.append({'item_support':name,'count':int(mask.sum()),'baseline_rmse':float(np.sqrt(np.square(prediction[mask]-truth[mask]).mean())) if mask.any() else None})
    out['datasets'][dataset]={'manifest':m,'density':m['n_ratings']/(m['n_users']*m['n_items']),
        'train_user_count_quantiles':dict(zip(['min','p25','median','p75','max'],np.quantile(counts,[0,.25,.5,.75,1]).tolist())),
        'val_positive_count_quantiles':dict(zip(['min','median','p90','max'],np.quantile(pos,[0,.5,.9,1]).tolist())),
        'validation_rating_mean':float(truth.mean()),'validation_rating_variance':float(truth.var()),'validation_baseline_rmse':float(np.sqrt(sse/len(truth))),'validation_baseline_r2':float(1-sse/sst),
        'validation_cold_item_count':int(cold.sum()),'validation_cold_item_fraction':float(cold.mean()),'baseline_item_support_cohorts':cohorts,
        'legacy_dense_float32_score_gib':m['n_users']*m['n_items']*4/1024**3,
        'legacy_eager_candidate_int64_gib_upper_bound':m['n_users']*m['n_items']*8/1024**3}
    del parts,train,val,truth,prediction,bu,bi;gc.collect()
for name in ['biasmf_rmse_v3','autorec_rmse_v3','multivae_v3']:
    rows=[]
    for seed in [17,29,43]:
        r=json.loads(Path(f'reports/v3/evidence/{name}_seed{seed}.json').read_text());trial=r['validation_trials'][r['selected_trial']]
        rows.append({'seed':seed,'best_epoch':trial['best_epoch'],'stopped_epoch':trial['stopped_epoch'],'max_epoch':r['config']['epochs'],'best_at_training_limit':trial['best_epoch']==r['config']['epochs']})
    out['v3_stopping'][name]=rows
Path('reports/v4/diagnosis.json').write_text(json.dumps(out,indent=2))
print(json.dumps({k:{x:v[x] for x in ['density','validation_baseline_rmse','validation_baseline_r2','validation_cold_item_fraction','legacy_dense_float32_score_gib','legacy_eager_candidate_int64_gib_upper_bound']} for k,v in out['datasets'].items()},indent=2))
