import argparse
from pathlib import Path
import yaml
from recranklab.experiments.hybrid import run_hybrid
p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['ml-1m','ml-10m']);p.add_argument('--name');p.add_argument('--smoke',action='store_true');p.add_argument('--plan',default='configs/v4/study.yaml');a=p.parse_args()
study=yaml.safe_load(Path(a.plan).read_text())
for dataset in (['ml-100k'] if a.smoke else ([a.dataset] if a.dataset else study['datasets'])):
    for name,spec in study['hybrids'].items():
        if a.name and name!=a.name:continue
        for seed in ([17] if a.smoke else study['seeds']):run_hybrid(dataset,name,spec,seed,study,output_root='outputs/v4_smoke' if a.smoke else 'outputs/v4')
