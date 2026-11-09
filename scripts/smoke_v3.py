"""Real ML-100K one-epoch pipelines, isolated from all headline outputs."""
from pathlib import Path
import yaml
from recranklab.utils.repro import config_load
from recranklab.experiments.run import run_one

study=yaml.safe_load(Path('configs/v3/study.yaml').read_text())
for name,spec in study['experiments'].items():
    cfg=config_load(spec['config'])
    cfg.update(dataset='ml-100k',epochs=1,trials=[cfg['trials'][0]],output_root='outputs/v3_smoke')
    run_one(spec['model'],cfg,17)
