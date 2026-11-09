"""Independent evidence/source/numerical audit; requires official ML-1M and ML-10M."""
import hashlib,json,subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from recranklab.data.prepare import digest,load
from recranklab.experiments.study_v4 import load_study_v4
from recranklab.utils.repro import fingerprint

ROOT=Path('reports/v4')

def source_yaml(sha,path):return yaml.safe_load(subprocess.check_output(['git','show',sha+':'+path],text=True))
def close(a,b):assert np.isclose(a,b,atol=1e-11,rtol=0),(a,b)


def main():
    manifest=json.loads((ROOT/'experiment_manifest.json').read_text());study=load_study_v4()
    assert manifest['schema_version']==4 and manifest['study']==study and manifest['status']['complete']
    assert len(manifest['runs'])==46 and manifest['status']['completed']==46
    for name,meta in manifest['artifacts'].items():assert digest(ROOT/name)==meta['sha256'],name
    for name,sha in manifest['immutable_historical'].items():assert digest(name)==sha,name
    data={};targets={};catalogs={};eligible={};sst={};counts={}
    refs=json.loads((ROOT/'rating_reference.json').read_text())
    for dataset in study['datasets']:
        parts,meta=load(dataset);data[dataset]=meta;assert refs[dataset]['manifest']==meta
        for stage in ('val','test'):
            truth=parts[stage].rating.to_numpy(dtype=float);ref=refs[dataset]['partitions'][stage]
            assert len(truth)==ref['count'];close(truth.mean(),ref['mean']);close(np.square(truth-truth.mean()).sum(),ref['sst'])
        y=parts['test'];sst[dataset]=refs[dataset]['partitions']['test']['sst'];counts[dataset]=y.groupby('user').size().to_numpy()
        catalog=np.sort(np.union1d(parts['train'].item,parts['val'].item));catalogs[dataset]=hashlib.sha256(catalog.astype('<i8').tobytes()).hexdigest()
        users=np.sort(y[(y.rating>=4)&y.item.isin(catalog)].user.unique());eligible[dataset]=hashlib.sha256(users.astype('<i8').tobytes()).hexdigest()
        targets[dataset]=(len(y),len(users),len(catalog))
        del parts,y
    rows=[];stats={};raw={};by_signature={}
    for key,entry in manifest['runs'].items():
        path=ROOT/entry['evidence'];assert digest(path)==entry['evidence_sha256'];r=json.loads(path.read_text());raw[key]=r;by_signature[r['signature']]=r
        env=r['environment'];sha=env['git_commit'];assert sha==entry['source_commit'] and not env['git_dirty']
        subprocess.run(['git','cat-file','-e',sha+'^{commit}'],check=True)
        assert r['signature']==entry['signature'];dataset=r['dataset'];name=r['experiment'];seed=r['seed'];assert r['data']==data[dataset]==entry['dataset']
        config=r['config'];metric=r.get('selection_metric',config.get('selection_metric'));trials=r['validation_trials']
        if entry['kind']=='base':
            spec=study['experiments'][name];expected=source_yaml(sha,'configs/base.yaml');expected.update(source_yaml(sha,spec['config']))
            phase=next(p for p,v in study['phases'].items() if name in v['experiments'])
            source_spec=source_yaml(sha,study['phases'][phase]['config'])['experiments'][name]
            expected.update(source_spec.get('dataset_overrides',{}).get(dataset,{}))
            expected.update(dataset=dataset,rating_min=.5 if dataset=='ml-10m' else 1.,model=spec['model'])
            assert expected==config,key
            assert fingerprint(config)==r['config_sha256']
            assert fingerprint({'config':config,'seed':seed,'model':r['model'],'commit':sha,'data':data[dataset]['fingerprint']})==r['signature']
            assert len(trials)==len(config['trials'])==2
            for i,t in enumerate(trials):
                assert t['overrides']==config['trials'][i] and 'error' not in t
                history=t.get('checkpoint_history')
                if history:
                    best=None
                    for record in history:
                        value=record['metrics'][metric]
                        if best is None or (value<best['metrics'][metric]-config['min_delta'] if metric=='rmse' else value>best['metrics'][metric]+config['min_delta']):best=record
                    assert best['epoch']==t['best_epoch'] and best['metrics']==t['metrics']
                    assert t['best_epoch']<=t['stopped_epoch']<=config['epochs']
            choose=min if metric=='rmse' else max;selected=choose(range(2),key=lambda i:trials[i]['metrics'][metric]);assert selected==r['selected_trial']
            selected_cfg={**config,**trials[selected]['overrides'],'seed':seed,'model':r['model']}
            if 'best_epoch' in trials[selected]:selected_cfg['epochs']=trials[selected]['best_epoch']
            assert selected_cfg==r['selected_config']
            if config.get('early_stopping'):assert len(r['final_loss_history'])==selected_cfg['epochs']
            if r['model']=='sgdmf':
                source=subprocess.check_output(['git','show',sha+':src/recranklab/models/sgd_kernel.cpp'])
                assert hashlib.sha256(source).hexdigest()==r['solver_diagnostics']['source_sha256']
            assert seed in spec.get('seeds',study['seeds'])
        else:
            spec=study['hybrids'][name]
            phase=next(p for p,v in study['phases'].items() if name in v['hybrids'])
            source_spec=source_yaml(sha,study['phases'][phase]['config'])['hybrids'][name];assert source_spec==spec
            assert config=={'dataset':dataset,'experiment':name,'seed':seed,'rating_min':.5 if dataset=='ml-10m' else 1.,**spec}
            assert len(trials)==len(spec['weights'])==5
            assert [x['weight_first'] for x in trials]==spec['weights']
            choose=min if metric=='rmse' else max;selected=choose(range(5),key=lambda i:trials[i]['metrics'][metric])
            assert selected==r['selected_trial'] and r['weight_first']==spec['weights'][selected]
            expected_sig=fingerprint({'config':config,'commit':sha,'data':data[dataset]['fingerprint'],'components':[x['signature'] for x in r['component_runs']]})
            assert expected_sig==r['signature']
            assert seed in study['seeds']
        score=r['test_metrics']
        for filename,f in entry['files'].items():
            assert digest(ROOT/f['path'])==f['sha256']
            if entry['kind']=='base':assert f['sha256']==r['per_user_rating_sha256' if filename=='per_user_rating.csv' else 'per_user_sha256']
            else:assert f['sha256']==r['files'][filename]
        if metric=='rmse':
            users=pd.read_csv(ROOT/entry['files']['per_user_rating.csv']['path']);assert users.evaluation_row.tolist()==list(range(len(users)))
            np.testing.assert_array_equal(users['count'],counts[dataset]);assert users['count'].sum()==score['rating_count']==targets[dataset][0]
            close(np.sqrt(users.sse.sum()/users['count'].sum()),score['rmse']);close(users.sae.sum()/users['count'].sum(),score['mae']);close(1-users.sse.sum()/sst[dataset],score['r2'])
            assert score['rmse_clipped_aux']<=score['rmse']+1e-12
        else:
            users=pd.read_csv(ROOT/entry['files']['per_user.csv']['path']);assert users.evaluation_row.tolist()==list(range(len(users)))
            assert len(users)==score['eligible_users']==targets[dataset][1]
            assert score['catalog_size']==targets[dataset][2] and score['eligible_user_sha256']==eligible[dataset] and score['candidate_catalog_sha256']==catalogs[dataset]
            for k in (10,20):
                for m in ('recall','ndcg'):
                    column=f'{m}@{k}';assert users[column].between(0,1+1e-12).all();close(users[column].mean(),score[column])
        stats[dataset,name,seed]=users;rows.append({'dataset':dataset,'experiment':name,'seed':seed,**{k:v for k,v in score.items() if isinstance(v,(float,int))}})
    # All selected/checkpoint dependency records must bind a published base run.
    for r in raw.values():
        for i,dependency in enumerate(r.get('component_runs',[])):
            base=by_signature[dependency['signature']];assert dependency['checkpoint_sha256']==base['checkpoint_sha256'] and dependency['source_commit']==base['environment']['git_commit']
            cache=r['validation_components'][i]
            assert cache['dependency_run_signature']==base['signature'] and cache['config']==base['selected_config']
            assert cache['signature']==fingerprint({'config':base['selected_config'],'source_commit':base['environment']['git_commit'],'data':base['data']['fingerprint']})
    frame=pd.DataFrame(rows);summary=pd.read_csv(ROOT/'results.csv').set_index(['dataset','experiment']);seeds=pd.read_csv(ROOT/'per_seed_results.csv')
    assert len(seeds)==46
    for (dataset,name),group in frame.groupby(['dataset','experiment']):
        spec=study['experiments'].get(name,{});assert sorted(group.seed)==spec.get('seeds',study['seeds'])
        assert summary.loc[(dataset,name),'seeds']==len(group)
        for metric in ('rmse','r2','mae','ndcg@10','ndcg@20','recall@10','recall@20'):
            if metric not in group or group[metric].isna().all():continue
            value=group[metric].dropna();close(value.mean(),summary.loc[(dataset,name),metric+'_mean']);close(value.std(ddof=1) if len(value)>1 else 0.,summary.loc[(dataset,name),metric+'_std'])
    for _,p in pd.read_csv(ROOT/'paired_differences.csv').iterrows():
        first,second=p.comparison.split(' - ');differences=[]
        for seed in study['seeds']:
            a=stats[p.dataset,first,seed]
            if second=='historical_autorec_v3':b=pd.read_csv(f'reports/v3/evidence/autorec_rmse_v3_seed{seed}_rating_users.csv')
            else:b=stats[p.dataset,second,17 if second in ('bias_baseline_v4','sparse_ease_v4') else seed]
            if p.metric=='rmse':
                np.testing.assert_array_equal(a['count'],b['count']);differences.append(np.sqrt(a.sse.sum()/a['count'].sum())-np.sqrt(b.sse.sum()/b['count'].sum()))
            else:differences.append((a[p.metric]-b[p.metric]).mean())
        close(np.mean(differences),p.mean_difference);assert p.ci_low<=p.ci_high and p.bootstrap_replicates==2000 and p.bootstrap_seed==2026
    assert len(json.loads((ROOT/'smoke_summary.json').read_text()))==10
    tracked=subprocess.check_output(['git','ls-files','data','outputs'],text=True);assert not tracked
    print('PASS: 46 V4 records across complete ML-1M/ML-10M; clean source/config/seed/data signatures, immutable history, stopping/refit, blend dependencies, official SST/R², per-user arithmetic, cohorts, aggregates and paired points.')

if __name__=='__main__':main()
