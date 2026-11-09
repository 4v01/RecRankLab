"""Execute the frozen cross-dataset study; final metrics never drive proposals."""
import argparse
from pathlib import Path
import yaml
from recranklab.experiments.run import run_one
from recranklab.utils.repro import config_load

p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['ml-1m','ml-10m']);p.add_argument('--model');p.add_argument('--smoke',action='store_true');a=p.parse_args()
study=yaml.safe_load(Path('configs/v4/study.yaml').read_text())
for dataset in ([a.dataset] if a.dataset else study['datasets']):
    for name,spec in study['experiments'].items():
        if a.model and a.model!=spec['model']:continue
        cfg=config_load(spec['config']);cfg['dataset']=dataset
        cfg['rating_min']=.5 if dataset=='ml-10m' else 1.
        seeds=spec.get('seeds',study['seeds'])
        if a.smoke:
            cfg.update(dataset='ml-100k',rating_min=1.,epochs=1,trials=[cfg['trials'][0]],output_root='outputs/v4_smoke');seeds=[17]
        for seed in seeds:run_one(spec['model'],cfg,seed)
    if a.smoke:break
