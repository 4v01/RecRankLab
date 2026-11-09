"""Frozen inference-context stress; parameters/targets/catalog/exclusion stay fixed."""
import argparse,gc,json
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from recranklab.data.prepare import load,digest
from recranklab.models.core import Recommender
from recranklab.evaluation.metrics import RankingProtocol,rating_metrics
from recranklab.utils.repro import seed_all,provenance


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--dataset',choices=['ml-1m','ml-10m']);a=parser.parse_args()
    selected=[a.dataset] if a.dataset else ['ml-1m','ml-10m']
    root=Path('reports/v4');rows=[];dependencies={};skipped=[];sources={}
    if (root/'context_stress_manifest.json').exists():
        previous=json.loads((root/'context_stress_manifest.json').read_text())
        rows=[r for r in pd.read_csv(root/'context_stress.csv').to_dict('records') if r['dataset'] not in selected]
        dependencies={k:v for k,v in previous['dependencies'].items() if k.split('/')[0] not in selected}
        sources={k:v for k,v in previous.get('evaluation_sources',{}).items() if k.split('/')[0] not in selected}
        skipped=[v for v in previous['resource_blocked'] if v['dataset'] not in selected]
    seed_all(2026,2)
    blocks=yaml.safe_load(Path('configs/v4/resource_blocks.yaml').read_text())['stress_records']
    source=provenance(require_clean=False)
    for dataset in selected:
        parts,data=load(dataset);history=pd.concat([parts['train'],parts['val']],ignore_index=True)
        for name in ('sparse_autorec_v4','sparse_ease_v4','history_tower_v4'):
            root=Path('outputs/v4')/dataset/name/'17'
            if not (root/'result.json').exists():
                declared=next((b for b in blocks if b['dataset']==dataset and b['experiment']==name),None)
                if declared is None:raise ValueError(f'Unexpected missing stress model: {dataset}/{name}')
                skipped.append(declared);continue
            result=json.loads((root/'result.json').read_text())
            assert digest(root/'model.pt')==result['checkpoint_sha256'];model=Recommender.load(root/'model.pt')
            sources[f'{dataset}/{name}']={'environment':source,'script_sha256':digest(__file__)}
            dependencies[f'{dataset}/{name}']={'result_signature':result['signature'],'source_commit':result['environment']['git_commit'],'checkpoint_sha256':result['checkpoint_sha256'],'data_fingerprint':data['fingerprint']}
            rng=np.random.default_rng(2026);is_tower=name=='history_tower_v4'
            if is_tower:
                original=model.history_ids.clone();valid=original.numpy()!=model.n_items;random=rng.random(original.shape);original_count=int(valid.sum())
            else:
                original=model.inputs.copy();random=rng.random(original.nnz);original_count=original.nnz
            protocol=RankingProtocol(history,parts['test'],data['n_users'],data['n_items']) if name!='sparse_autorec_v4' else None
            baseline=None
            for drop in (0.,.2,.5,.8):
                if is_tower:
                    import torch
                    model.history_ids=original.clone();hidden=valid&(random<drop);model.history_ids[torch.from_numpy(hidden)]=model.n_items;remaining=original_count-int(hidden.sum())
                else:
                    model.inputs=original.copy();model.inputs.data[random<drop]=0.;model.inputs.eliminate_zeros();remaining=model.inputs.nnz
                model.cache=None
                scores=rating_metrics(model,parts['test']) if protocol is None else protocol.evaluate(model,batch_size=128)[0]
                metric='rmse' if protocol is None else 'ndcg@10'
                if drop==0.:
                    assert np.isclose(scores[metric],result['test_metrics'][metric],atol=1e-11,rtol=0)
                    baseline=scores[metric]
                row={'dataset':dataset,'experiment':name,'model_seed':17,'perturbation_seed':2026,'nominal_context_drop':drop,'original_context_entries':original_count,'remaining_context_entries':remaining,'actual_context_drop':1-remaining/original_count,'primary_metric':metric,'primary_value':scores[metric],'difference_from_full_context':scores[metric]-baseline,**scores}
                rows.append(row);print(dataset,name,drop,scores[metric],flush=True)
            del model,protocol,original;gc.collect()
        del parts,history;gc.collect()
    root=Path('reports/v4');pd.DataFrame(rows).to_csv(root/'context_stress.csv',index=False)
    metadata={'completed':len(rows),'expected':24,'evaluation_sources':sources,'resource_blocked':skipped,'scope':'fixed fitted parameters; nested inference-input missingness; complete original catalog and exclusion masks; no retraining/selection','evaluation_source':source,'evaluation_script_sha256':digest(__file__),'fractions':[0.,.2,.5,.8],'dependencies':dependencies}
    (root/'context_stress_manifest.json').write_text(json.dumps(metadata,indent=2))

if __name__=='__main__':main()
