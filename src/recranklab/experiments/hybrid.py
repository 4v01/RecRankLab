"""Frozen convex blends with validation-only selection and immutable dependencies."""
import gc,json,time
from pathlib import Path
import numpy as np
import pandas as pd
from recranklab.data.prepare import load,digest
from recranklab.evaluation.metrics import RankingProtocol,rating_metrics,rating_user_errors
from recranklab.models.core import create,Recommender
from recranklab.utils.repro import provenance,seed_all,fingerprint


class RatingBlend:
    rating_model=True
    def __init__(self,first,second,weight,config):
        self.first,self.second,self.weight,self.config=first,second,weight,config
        self.user_support=first.user_support;self.item_support=first.item_support
    def predict_pairs(self,users,items):
        return self.weight*self.first.predict_pairs(users,items)+(1-self.weight)*self.second.predict_pairs(users,items)
    def predict_scores(self,users,items=None):
        return self.weight*self.first.predict_scores(users,items)+(1-self.weight)*self.second.predict_scores(users,items)


class RankingBlend:
    def __init__(self,first,second,weight,history):
        self.first,self.second,self.weight=first,second,weight
        self.catalog=np.sort(history.item.unique())
    def predict_scores(self,users,items=None):
        first=self.first.predict_scores(users,self.catalog);second=self.second.predict_scores(users,self.catalog)
        def normalize(value):return (value-value.mean(1,keepdims=True))/value.std(1,keepdims=True).clip(min=1e-12)
        value=self.weight*normalize(first)+(1-self.weight)*normalize(second)
        selected=self.catalog if items is None else np.asarray(items)
        return value[:,np.searchsorted(self.catalog,selected)]


def validation_component(result,parts):
    cfg=result['selected_config'];key=fingerprint({'config':cfg,'source_commit':result['environment']['git_commit'],'data':result['data']['fingerprint']})
    root=Path('outputs/v4_validation_components')/result['dataset']/result['experiment']/str(result['seed'])
    root.mkdir(parents=True,exist_ok=True);path=root/'model.pt';meta=root/'metadata.json'
    if path.exists() and meta.exists():
        saved=json.loads(meta.read_text())
        if saved['signature']!=key or digest(path)!=saved['checkpoint_sha256']:raise ValueError('Validation component cache mismatch')
        return Recommender.load(path),saved
    seed_all(result['seed'],cfg['threads']);begin=time.perf_counter()
    model=create(result['model'],result['data']['n_users'],result['data']['n_items'],cfg).fit(parts['train'])
    model.save(path)
    saved={'signature':key,'checkpoint_sha256':digest(path),'config':cfg,'dependency_run_signature':result['signature'],'fit_seconds':time.perf_counter()-begin}
    meta.write_text(json.dumps(saved,indent=2))
    return model,saved


def run_hybrid(dataset,name,spec,seed,study,output_root="outputs/v4"):
    begin=time.perf_counter();env=provenance();parts,data=load(dataset)
    config={'dataset':dataset,'experiment':name,'seed':seed,'rating_min':.5 if dataset=='ml-10m' else 1.,**spec}
    source=[]
    for component in spec['components']:
        component_seed=seed if component not in ('sparse_ease_v4','bias_baseline_v4') else 17
        root=Path(output_root)/dataset/component/str(component_seed)
        saved=json.loads((root/'result.json').read_text())
        if digest(root/'model.pt')!=saved['checkpoint_sha256']:raise ValueError('Final component checkpoint mismatch')
        source.append((root,saved))
    signature=fingerprint({'config':config,'commit':env['git_commit'],'data':data['fingerprint'],'components':[x[1]['signature'] for x in source]})
    out=Path(output_root)/dataset/name/str(seed);out.mkdir(parents=True,exist_ok=True)
    if (out/'result.json').exists():
        saved=json.loads((out/'result.json').read_text())
        if saved['signature']!=signature:raise ValueError('Refusing hybrid overwrite')
        return saved
    first,first_meta=validation_component(source[0][1],parts);second,second_meta=validation_component(source[1][1],parts)
    records=[];metric=spec['objective']
    protocol=RankingProtocol(parts['train'],parts['val'],data['n_users'],data['n_items']) if metric!='rmse' else None
    for weight in spec['weights']:
        model=RatingBlend(first,second,weight,config) if metric=='rmse' else RankingBlend(first,second,weight,parts['train'])
        scores=rating_metrics(model,parts['val']) if metric=='rmse' else protocol.evaluate(model,batch_size=128)[0]
        records.append({'weight_first':weight,'metrics':scores});print(f'{dataset} {name} seed={seed} weight={weight} val_{metric}={scores[metric]:.6f}',flush=True)
    choose=min if metric=='rmse' else max
    selected=choose(range(len(records)),key=lambda i:records[i]['metrics'][metric]);weight=records[selected]['weight_first']
    selection={'signature':signature,'selected_trial':selected,'weight_first':weight,'selection_metric':metric,'validation_trials':records,'validation_components':[first_meta,second_meta]}
    (out/'selection.json').write_text(json.dumps(selection,indent=2))
    del model,first,second,protocol;gc.collect()
    first=Recommender.load(source[0][0]/'model.pt');second=Recommender.load(source[1][0]/'model.pt')
    final_train=pd.concat([parts['train'],parts['val']],ignore_index=True)
    if metric=='rmse':
        model=RatingBlend(first,second,weight,config);scores=rating_metrics(model,parts['test'])
        users=rating_user_errors(model,parts['test']).drop(columns='user').rename_axis('evaluation_row').reset_index()
        users.to_csv(out/'per_user_rating.csv',index=False);files={'per_user_rating.csv':digest(out/'per_user_rating.csv')}
    else:
        model=RankingBlend(first,second,weight,final_train)
        scores,users=RankingProtocol(final_train,parts['test'],data['n_users'],data['n_items']).evaluate(model,batch_size=128)
        users.drop(columns='user').rename_axis('evaluation_row').reset_index().to_csv(out/'per_user.csv',index=False);files={'per_user.csv':digest(out/'per_user.csv')}
    result={'schema_version':1,'signature':signature,'dataset':dataset,'experiment':name,'model':name,'seed':seed,'config':config,'environment':env,'data':data,**selection,'test_metrics':scores,'component_runs':[{'signature':r['signature'],'source_commit':r['environment']['git_commit'],'path':str(root),'checkpoint_sha256':r['checkpoint_sha256']} for root,r in source],'files':files,'total_seconds':time.perf_counter()-begin}
    (out/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False));print(f'FINAL {dataset} {name} seed={seed}: {scores}',flush=True)
    return result
