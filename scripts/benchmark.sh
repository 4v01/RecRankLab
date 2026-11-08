#!/usr/bin/env bash
set -euo pipefail
python -m recranklab.data.prepare --dataset ml-1m --config configs/base.yaml
python -m recranklab.data.prepare --dataset ml-100k --config configs/base.yaml
for model in popularity biasmf autorec bpr_mf lightgcn ease; do
  config="configs/${model}.yaml"
  if [[ "$model" == popularity ]]; then config=configs/base.yaml; fi
  python -m recranklab.experiments.run --model "$model" --config "$config" --smoke --seeds 17
done
python scripts/run_study.py
python scripts/data_audit.py
python -m recranklab.experiments.summarize --output reports/results.csv
python scripts/audit_results.py
