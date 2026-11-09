"""Execute the frozen v3 study; no test-guided proposals or overwrites."""
from pathlib import Path
import subprocess
import sys
import yaml
study=yaml.safe_load(Path('configs/v3/study.yaml').read_text())
for name,spec in study['experiments'].items():
    subprocess.run([sys.executable,'-m','recranklab.experiments.run','--model',spec['model'],
                    '--config',spec['config']],check=True)
