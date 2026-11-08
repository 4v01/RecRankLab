# RecRankLab

Reproducible MovieLens rating prediction and full-catalog recommendation research. This v2 continues the supplied review package and compares Popularity, BiasMF, user AutoRec, BPR-MF and LightGCN, then tests item AutoRec, larger-budget BPR/LightGCN, EASE and bounded validation-memory search. It does not require advanced models to win.

## Reproduction

Python 3.12, Linux CPU. Full Git history is required to audit experimental source commits. Read the [MovieLens terms](https://grouplens.org/datasets/movielens/) before downloading. Data/checkpoints are never redistributed.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-lock.txt
pip install --no-deps -e .
pytest -q
python -m recranklab.data.prepare --dataset ml-1m --config configs/base.yaml
python -m recranklab.data.prepare --dataset ml-100k --config configs/base.yaml
python -m recranklab.experiments.run --model popularity --config configs/base.yaml
python -m recranklab.experiments.run --model autorec --config configs/autorec.yaml
python -m recranklab.experiments.run --model bpr_mf --config configs/bpr_mf.yaml
python -m recranklab.experiments.run --model lightgcn --config configs/lightgcn.yaml
python -m recranklab.experiments.run --model ease --config configs/ease_selfopt.yaml
# Complete preregistered 3-seed study, including BiasMF, improved methods and ablations:
python scripts/run_study.py
python scripts/data_audit.py
python -m recranklab.experiments.summarize --output reports/results.csv
python scripts/audit_results.py
```

`bash scripts/benchmark.sh` additionally runs six real ML-100K CPU smoke checks. Individual smoke: `python -m recranklab.experiments.run --model lightgcn --config configs/lightgcn.yaml --smoke --seeds 17`. Smoke metrics are separate from research results.

Commit source/config before training. The runner rejects dirty trees, saves all validation trials and persists selection before refitting train+validation and evaluating test once. Completed matching signatures are reused; incompatible overwrites are rejected. For a new experiment, commit a new output_root in a new config. `--allow-dirty` is development-only and cannot produce accepted report evidence. To reproduce an existing result, check out its exact source commit in the manifest before running it; later documentation commits intentionally change HEAD.

## Fair evaluation

Per-user temporal 80/10/10, deterministic ties; positives are ratings >=4. Full fit-supported catalog; mask all prior observations including low ratings. Shared evaluator, eligible-user set and candidate policy; report Recall/NDCG@10/20, coverage and cold exclusions. Explicit models report unclipped observed-rating RMSE/MAE. Negative sampling uses only fit data; blacklisting future positives would leak future behavior. This is not globally forward-in-time evaluation or click feedback.

The original small-budget core comparison has two validation learning rates and 20 epochs. Improvement models have four candidates and explicitly larger training budgets. The EASE grid vs memory-search ablation has exactly four validation fits each. Select by validation NDCG@10 per seed, reinitialize for final fit, test once. Report actual runtime rather than claiming equal compute. Seeds: 17,29,43; deterministic EASE repetitions are reproducibility checks. Prior v1 test results were known when proposing this follow-up; new numbers are still from new runs under the unchanged evaluator.

## Research and review artifacts

- [Technical report](reports/technical_report.md), [results](reports/results.csv), [per-seed evidence](reports/per_seed_results.csv)
- [Frozen methods](reports/methodology.md), [limitations](reports/limitations.md), [study plan](configs/study.yaml)
- [Ablations](reports/ablations.csv), [all validation candidates](reports/validation_trials.csv), [paired bootstrap](reports/paired_differences.csv)
- [Manifest](reports/experiment_manifest.json), [completion status](reports/completion_status.json), [data audit](reports/data_audit.json), [local/CI validation](reports/validation.md)
- [Historical v1 package](reports/legacy_v1): preserved as supplied, excluded from all new result tables; its original Git history is unavailable and not independently attested.

Each new result includes source SHA, configs, data/split checksums, seed, timestamp, hardware/software and anonymous user-level metrics. Report-generation provenance is separate. Raw/processed data and model checkpoints remain ignored. `scripts/audit_results.py` independently verifies evidence hashes, selection, numerical aggregates and source history. GitHub Actions runs unit tests, the artifact audit and real-data CPU smoke checks without paid APIs/GPU; it uploads only JSON smoke metrics.

## Bounded self-optimization

The EASE extension fits binary positive histories with zero-diagonal ridge item regression. Its memory search starts with two ridge anchors, then makes bounded multiplicative proposals around the validation elite, retaining failed as well as successful trials. It cannot access a test evaluator. The exact four-fit policy is fixed before experimentation. It is a small hypothesis/validation/memory mechanism inspired by [CogAlpha](https://graceim.ai/zh/research/cognitive-alpha-mining) and [XALPHA](https://graceim.ai/zh/research/xalpha-memory-driven-ai-quant-researcher), not a reproduction, affiliation or claim about financial performance. No LLM or arbitrary generated code runs in this module. Optional LLM reranking is deferred.

## API and references

`create(name, n_users, n_items, config)` returns `fit(training_frame)`, `predict_scores(user_ids, item_ids=None)` (Cartesian scores, higher better), `save(path)`, `Recommender.load(path)` with weights-only deserialization. Fit frames contain only user/item/rating. Item/user variants and samplers reconstruct training-only context from local checkpoints. EASE has no gradient-trained parameters but stores O(items²) coefficients; dense scoring and histories also limit scale.

Original code uses MIT; MovieLens has separate terms. References: [AutoRec](https://doi.org/10.1145/2740908.2742726), [BPR](https://proceedings.mlr.press/r7/rendle09a.html), [LightGCN](https://arxiv.org/abs/2002.02126), [EASE](https://doi.org/10.1145/3308558.3313710). No RecRanker code or historical AutoRec RMSE 0.13 claim is used.
