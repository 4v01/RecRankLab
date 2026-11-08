"""Run the preregistered study, resuming completed signatures without re-testing."""
from pathlib import Path
import subprocess
import sys
import yaml

study=yaml.safe_load(Path('configs/study.yaml').read_text())
for experiment, entry in study['experiments'].items():
    subprocess.run([sys.executable,'-m','recranklab.experiments.run', '--model',entry['model'],
                    '--config',entry['config']],check=True)
