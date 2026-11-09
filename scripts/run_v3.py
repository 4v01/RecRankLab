"""Run declared phases on a fresh checkout, with immutable output signatures."""
import argparse
import subprocess
import sys
from recranklab.experiments.study import load_study
p=argparse.ArgumentParser();p.add_argument('--phase',choices=['all','primary','followups'],default='all');a=p.parse_args()
study=load_study();primary=set(study['phases']['primary'])
for name,spec in study['experiments'].items():
    if a.phase=='primary' and name not in primary:continue
    if a.phase=='followups' and name in primary:continue
    subprocess.run([sys.executable,'-m','recranklab.experiments.run','--model',spec['model'],
                    '--config',spec['config']],check=True)
