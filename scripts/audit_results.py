"""Independent artifact, selection, arithmetic, source and population audit."""
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
import pandas as pd
import yaml


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def fingerprint(value):return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


def main():
    root=Path('reports');manifest=json.loads((root/'experiment_manifest.json').read_text())
    assert manifest['schema_version']==2
    assert manifest['status']['complete'], 'Frozen study is incomplete; see completion_status.json'
    study=yaml.safe_load(Path('configs/study.yaml').read_text())
    assert study==manifest['study']
    for name,entry in manifest['artifacts'].items():
        assert sha(root/name)==entry['sha256'],f'Artifact hash mismatch: {name}'
        assert all(k in manifest['runs'] for k in entry['source_runs'])
    results=pd.read_csv(root/'results.csv').set_index('experiment')
    records={};populations=set();per_users={}
    for key,entry in manifest['runs'].items():
        run=json.loads((root/entry['evidence']).read_text())
        for field in ['model','seed','config','selected_config','config_sha256','data','environment']:
            assert run[field]==entry[field],(key,field)
        assert not run['environment']['git_dirty']
        subprocess.run(['git','cat-file','-e',entry['environment']['git_commit']+'^{commit}'],check=True)
        assert fingerprint(run['config'])==run['config_sha256']
        assert fingerprint({'config':run['config'],'seed':run['seed'],'model':run['model'],
               'commit':run['environment']['git_commit'],'data':run['data']['fingerprint']})==run['signature']
        # Independently validate the dataset manifest's self fingerprint.
        data=dict(run['data']);finger=data.pop('fingerprint');assert fingerprint(data)==finger
        if key.startswith('smoke_'):
            assert run['dataset']=='ml-100k' and run['selected_config']['epochs']==1
            continue
        name=run['experiment'];plan=study['experiments'][name]
        assert run['model']==plan['model'] and len(run['validation_trials'])==plan['trials']
        trials=run['validation_trials'];selected=max(range(len(trials)),key=lambda j:trials[j]['metrics']['ndcg@10'])
        assert selected==run['selected_trial']
        expected={**run['config'],**trials[selected]['overrides'],'seed':run['seed'],'model':run['model']}
        assert expected==run['selected_config']
        users=root/'evidence'/f'{key}_users.csv';assert sha(users)==run['per_user_sha256']
        per=pd.read_csv(users);per_users[name,run['seed']]=per
        assert per.evaluation_row.tolist()==list(range(len(per)))
        assert len(per)==run['test_metrics']['eligible_users']
        for metric in ['recall@10','ndcg@10','recall@20','ndcg@20']:
            assert np.isclose(per[metric].mean(),run['test_metrics'][metric],atol=1e-12,rtol=0)
            assert per[metric].between(0,1).all()
        metric=run['test_metrics']
        populations.add((metric['eligible_user_sha256'],metric['candidate_catalog_sha256'],run['data']['fingerprint']))
        records.setdefault(name,[]).append(run)
    assert len(populations)==1
    assert set(records)==set(study['experiments'])==set(results.index)
    for name,runs in records.items():
        assert sorted(r['seed'] for r in runs)==study['seeds']
        for metric in ['recall@10','ndcg@10','recall@20','ndcg@20','coverage@10','coverage@20','rmse','mae']:
            if metric not in runs[0]['test_metrics']:continue
            values=[r['test_metrics'][metric] for r in runs]
            assert np.isclose(np.mean(values),results.loc[name,metric+'_mean'],atol=1e-12,rtol=0)
            assert np.isclose(np.std(values,ddof=1),results.loc[name,metric+'_std'],atol=1e-12,rtol=0)
    paired=pd.read_csv(root/'paired_differences.csv')
    for row in paired.to_dict('records'):
        a,b=row['comparison'].split(' - ');m=row['metric']
        diffs=[(per_users[a,s][m]-per_users[b,s][m]).to_numpy() for s in study['seeds']]
        assert np.isclose(np.mean(diffs),row['mean_difference'],atol=1e-12,rtol=0)
        assert row['ci_low']<=row['mean_difference']<=row['ci_high']
    tracked=subprocess.check_output(['git','ls-files'],text=True).splitlines()
    assert not any(p.startswith(('data/','outputs/')) or p.endswith(('.pt','.npz')) for p in tracked)
    print(f'PASS: {len(records)} experiments × 3 seeds; hashes, selection, means/SD, paired differences, source history, shared populations and data exclusions.')


if __name__=='__main__':main()
