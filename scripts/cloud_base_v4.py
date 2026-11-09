"""One frozen ML-10M base case per hosted CPU job; upload no rating contexts."""
import argparse,json,gc
from pathlib import Path
from recranklab.data.prepare import load
from recranklab.experiments.run import run_one
from recranklab.experiments.hybrid import validation_component
from recranklab.utils.repro import config_load
p=argparse.ArgumentParser();p.add_argument('--model',choices=['sgdmf','mf_dcn','sparse_ease','history_tower'],required=True);p.add_argument('--seed',type=int,required=True);a=p.parse_args()
cfg=config_load(f'configs/v4/{a.model}.yaml');cfg.update(dataset='ml-10m',rating_min=.5)
r=run_one(a.model,cfg,a.seed)
gc.collect();parts,_=load('ml-10m')
model,metadata=validation_component(r,parts)
print('Validation-only selected component saved for frozen blend:',metadata['checkpoint_sha256'],flush=True)
