"""Real ML-100K one-epoch checks; extra phases can be checked separately."""
import argparse
from recranklab.utils.repro import config_load
from recranklab.experiments.run import run_one
from recranklab.experiments.study import load_study
p=argparse.ArgumentParser();p.add_argument('--phase',choices=['all','primary','followups'],default='all');a=p.parse_args()
study=load_study();primary=set(study['phases']['primary'])
for name,spec in study['experiments'].items():
    if a.phase=='primary' and name not in primary:continue
    if a.phase=='followups' and name in primary:continue
    cfg=config_load(spec['config'])
    cfg.update(dataset='ml-100k',epochs=1,trials=[cfg['trials'][0]],output_root='outputs/v3_smoke')
    run_one(spec['model'],cfg,17)
