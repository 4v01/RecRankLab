"""Audit genuine v3 results independently of the report generator."""
import hashlib,json,subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import yaml


def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def fingerprint(x):return hashlib.sha256(json.dumps(x,sort_keys=True).encode()).hexdigest()


def main():
    root=Path('reports/v3');m=json.loads((root/'experiment_manifest.json').read_text())
    assert m['schema_version']==3 and m['status']['complete']
    plan=yaml.safe_load(Path('configs/v3/study.yaml').read_text());assert m['study']==plan
    for name,entry in m['artifacts'].items():assert digest(root/name)==entry['sha256'],name
    assert digest(m['historical_reference']['path'])==m['historical_reference']['sha256']
    assert digest('reports/experiment_manifest.json')==m['historical_reference']['manifest_sha256']
    population=set();groups={};rating_users={};ranking_users={}
    for key,entry in m['runs'].items():
        r=json.loads((root/entry['evidence']).read_text());name=r['experiment'];seed=r['seed']
        assert r['environment']==entry['environment'] and not r['environment']['git_dirty']
        sha=r['environment']['git_commit'];assert sha==entry['source_commit']
        subprocess.run(['git','cat-file','-e',sha+'^{commit}'],check=True)
        assert fingerprint(r['config'])==r['config_sha256']
        assert fingerprint({'config':r['config'],'seed':seed,'model':r['model'],'commit':sha,'data':r['data']['fingerprint']})==r['signature']
        assert r['selected_config']==entry['selected_config'] and r['data']==entry['dataset']
        assert len(r['validation_trials'])==plan['experiments'][name]['trials']
        objective=plan['experiments'][name]['objective'];assert r['config'].get('selection_metric','ndcg@10')==objective
        valid=[i for i,t in enumerate(r['validation_trials']) if 'error' not in t]
        choose=min if objective=='rmse' else max
        selected=choose(valid,key=lambda i:r['validation_trials'][i]['metrics'][objective]);assert selected==r['selected_trial']
        t=r['validation_trials'][selected]
        expected={**r['config'],**t['overrides'],'seed':seed,'model':r['model']}
        if 'best_epoch' in t:expected['epochs']=t['best_epoch']
        assert expected==r['selected_config']
        for trial in r['validation_trials']:
            if 'error' in trial:continue
            if 'checkpoint_history' in trial:
                history=trial['checkpoint_history'];best=None;delta=r['config'].get('min_delta',0.)
                assert [c['epoch'] for c in history]==sorted(set(c['epoch'] for c in history))
                for current in history:
                    val=current['metrics'][objective]
                    if best is None or (val<best['metrics'][objective]-delta if objective=='rmse' else val>best['metrics'][objective]+delta):best=current
                assert best['epoch']==trial['best_epoch'] and best['metrics']==trial['metrics']
                assert trial['best_epoch']<=trial['stopped_epoch']<=r['config']['epochs']
            if 'solver_diagnostics' in trial:assert trial['solver_diagnostics']['nonconverged_columns']==0
        score=r['test_metrics'];population.add((r['data']['fingerprint'],score['eligible_user_sha256'],score['candidate_catalog_sha256']))
        p=root/'evidence'/f'{key}_users.csv';assert digest(p)==r['per_user_sha256'];users=pd.read_csv(p)
        assert len(users)==score['eligible_users'] and users.evaluation_row.tolist()==list(range(len(users)))
        ranking_users[name,seed]=users
        for metric in ['ndcg@10','recall@10','ndcg@20','recall@20']:
            assert np.isclose(users[metric].mean(),score[metric],atol=1e-12,rtol=0)
        if objective=='rmse':
            p=root/'evidence'/f'{key}_rating_users.csv';assert digest(p)==r['per_user_rating_sha256']
            stats=pd.read_csv(p);rating_users[name,seed]=stats
            assert stats.evaluation_row.tolist()==list(range(len(stats))) and stats['count'].sum()==score['rating_count']
            assert np.isclose(np.sqrt(stats.sse.sum()/stats['count'].sum()),score['rmse'],atol=1e-12,rtol=0)
            assert np.isclose(stats.sae.sum()/stats['count'].sum(),score['mae'],atol=1e-12,rtol=0)
            assert score['rmse_clipped_aux']<=score['rmse']+1e-12
        if r['model']=='slim':assert r['solver_diagnostics']['nonconverged_columns']==0
        groups.setdefault(name,[]).append(r)
    assert len(population)==1 and set(groups)==set(plan['experiments'])
    results=pd.read_csv(root/'results.csv').set_index('experiment')
    for name,runs in groups.items():
        assert sorted(r['seed'] for r in runs)==plan['seeds']
        for metric in ['rmse','mae','ndcg@10','recall@10','ndcg@20','recall@20']:
            if metric not in runs[0]['test_metrics']:continue
            vals=[r['test_metrics'][metric] for r in runs]
            assert np.isclose(np.mean(vals),results.loc[name,metric+'_mean'],atol=1e-12,rtol=0)
            assert np.isclose(np.std(vals,ddof=1),results.loc[name,metric+'_std'],atol=1e-12,rtol=0)
    old=pd.read_csv('reports/results.csv').set_index('experiment')
    assert np.isclose(results.loc['ease_v3','ndcg@10_mean'],old.loc['ease','ndcg@10_mean'],atol=1e-12,rtol=0)
    comparisons=pd.read_csv(root/'paired_differences.csv')
    for row in comparisons.to_dict('records'):
        a,b=row['comparison'].split(' - ');metric=row['metric']
        if metric=='rmse':
            values=[]
            for seed in plan['seeds']:
                x,y=rating_users[a,seed],rating_users[b,seed]
                assert x.evaluation_row.equals(y.evaluation_row) and x['count'].equals(y['count'])
                values.append(np.sqrt(x.sse.sum()/x['count'].sum())-np.sqrt(y.sse.sum()/y['count'].sum()))
            delta=np.mean(values)
        else:delta=np.mean([(ranking_users[a,s][metric]-ranking_users[b,s][metric]).to_numpy() for s in plan['seeds']])
        assert np.isclose(delta,row['mean_difference'],atol=1e-12,rtol=0)
    tracked=subprocess.check_output(['git','ls-files'],text=True).splitlines()
    assert not any(p.startswith(('data/','outputs/')) or p.endswith(('.pt','.npz')) for p in tracked)
    print(f'PASS: {len(groups)} v3 experiments × 3 seeds; immutable hashes, source/config/data, stopping/selection/refit epochs, per-user rating/ranking arithmetic, paired differences, convergence and unchanged EASE reference.')


if __name__=='__main__':main()
