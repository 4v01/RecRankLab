import hashlib
import importlib.metadata
import json
import os
import platform
import random
import subprocess
from pathlib import Path

import numpy as np
import torch
import yaml
from threadpoolctl import threadpool_limits, threadpool_info


def seed_all(seed, threads=2):
    threadpool_limits(limits=threads)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)


def config_load(path):
    cfg = yaml.safe_load(Path("configs/base.yaml").read_text())
    cfg.update(yaml.safe_load(Path(path).read_text()))
    return cfg


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def provenance(require_clean=True):
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
    if require_clean and dirty:
        raise RuntimeError("Commit code/config changes before running experiments")
    result={"git_commit": sha, "git_dirty": bool(dirty), "python": platform.python_version(),
            "platform": platform.platform(), "cpu": next((x.split(':', 1)[1].strip() for x in Path('/proc/cpuinfo').read_text().splitlines() if x.startswith('model name')), platform.processor()),
            "logical_cpus": os.cpu_count(), "device": "cpu", "blas": threadpool_info(),
            "packages": {p: importlib.metadata.version(p) for p in ("torch", "numpy", "pandas", "scipy", "scikit-learn", "PyYAML", "matplotlib", "pytest")}}
    if os.environ.get("GITHUB_RUN_ID"):
        result["execution"]={"provider":"github_actions","run_id":os.environ["GITHUB_RUN_ID"],
            "run_attempt":os.environ.get("GITHUB_RUN_ATTEMPT"),"workflow_sha":os.environ.get("GITHUB_WORKFLOW_SHA"),
            "job":os.environ.get("GITHUB_JOB"),"runner_os":os.environ.get("RUNNER_OS"),
            "url":f"https://github.com/{os.environ.get('GITHUB_REPOSITORY')}/actions/runs/{os.environ['GITHUB_RUN_ID']}"}
    return result
