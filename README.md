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
- [Chinese review handoff](reports/handoff_zh.md), [GitHub data/method/result benchmark](reports/external_benchmark/README.md), [v3 upgrade plan](reports/external_benchmark/upgrade_plan.md)

Each new result includes source SHA, configs, data/split checksums, seed, timestamp, hardware/software and anonymous user-level metrics. Report-generation provenance is separate. Raw/processed data and model checkpoints remain ignored. `scripts/audit_results.py` independently verifies evidence hashes, selection, numerical aggregates and source history. GitHub Actions runs unit tests, the artifact audit and real-data CPU smoke checks without paid APIs/GPU; it uploads only JSON smoke metrics.

The external review covers six pinned source snapshots and 30 source-reported values. These are not local reproductions and are not pooled with our experimental table. Its separate manifest records citation/file hashes; `python reports/external_benchmark/verify.py` audits its bookkeeping and synthetic metric counterexamples. The original run commits behind upstream notebook/paper claims are not established by pinning a review snapshot.

## Bounded self-optimization

The EASE extension fits binary positive histories with zero-diagonal ridge item regression. Its memory search starts with two ridge anchors, then makes bounded multiplicative proposals around the validation elite, retaining failed as well as successful trials. It cannot access a test evaluator. The exact four-fit policy is fixed before experimentation. It is a small hypothesis/validation/memory mechanism inspired by [CogAlpha](https://graceim.ai/zh/research/cognitive-alpha-mining) and [XALPHA](https://graceim.ai/zh/research/xalpha-memory-driven-ai-quant-researcher), not a reproduction, affiliation or claim about financial performance. No LLM or arbitrary generated code runs in this module. Optional LLM reranking is deferred.

## API and references

`create(name, n_users, n_items, config)` returns `fit(training_frame)`, `predict_scores(user_ids, item_ids=None)` (Cartesian scores, higher better), `save(path)`, `Recommender.load(path)` with weights-only deserialization. Fit frames contain only user/item/rating. Item/user variants and samplers reconstruct training-only context from local checkpoints. EASE has no gradient-trained parameters but stores O(items²) coefficients; dense scoring and histories also limit scale.

Original code uses MIT; MovieLens has separate terms. References: [AutoRec](https://doi.org/10.1145/2740908.2742726), [BPR](https://proceedings.mlr.press/r7/rendle09a.html), [LightGCN](https://arxiv.org/abs/2002.02126), [EASE](https://doi.org/10.1145/3308558.3313710). No RecRanker code or historical AutoRec RMSE 0.13 claim is used.

## V3: rating-oriented selection and strong ranking baselines

V3 separates validation-RMSE-selected rating models from validation-NDCG-selected ranking models. It adds five-epoch checkpoint checks, patience3, clean reinitialization/refit at the selected epoch, stronger MF regularization, observed-only bias-residual AutoRec, cosine ItemKNN, RP3beta, positive ElasticNet SLIM and multinomial MultiVAE. All retain the original temporal split, full-catalog evaluator and unclipped primary rating metrics. Source/configuration is committed before the v3 study. Historical v2 evidence remains unchanged.

```bash
python scripts/run_v3.py
python scripts/smoke_v3.py
python scripts/report_v3.py
python scripts/audit_v3.py
```

Use the exact source commit in the [v3 manifest](reports/v3/experiment_manifest.json) when reproducing existing signatures. The [frozen methodology](reports/v3/methodology.md), [new rating-source inspection](reports/v3/research/README.md) and [v3 actual-results report](reports/v3/technical_report.md) distinguish rating accuracy, ranking and external reports. In particular, some lower external RMSE values use random splits/clipped predictions, and an inspected AutoRec implementation reconstructs its test input; none of these changes are imported as a shortcut to lower primary RMSE.

V3 has three explicitly separate phases: 24 primary final runs, 3 low-decay follow-up runs and 3 SVD++ rating-reference runs. Additional candidate and validation-checkpoint budgets are disclosed. R² is now included for observed-rating predictions, alongside primary unclipped RMSE/MAE; finished raw records are preserved and missing historical R² is derived from frozen SSE/count and the same official test SST.

For strict reproduction of delivered signatures, use the source SHA in each run: `964084262c25184b2ca363bc5ea5022887e16d79` for the original primary script, `5b8bd6a86adcd7881c36282956436cf0b0362444` with `--model autorec --config configs/v3/autorec_lowreg.yaml`, and `49cda6e052b1fbed5c45a2eba60f91ffb0f6a6aa` with `--model svdpp --config configs/v3/svdpp.yaml`. The current runner offers `--phase primary` / `--phase followups` / default `all` for new runs. The R² source-data audit needs `python -m recranklab.data.prepare --dataset ml-1m --config configs/base.yaml` first.

V3 delivery: [Chinese review handoff](reports/v3/handoff_zh.md), [validation evidence](reports/v3/validation.md), [rating RMSE/MAE/R²](reports/v3/rating_results.csv) and [ranking results](reports/v3/ranking_results.csv). The best mean RMSE is 0.869547 (R² 0.424549), a small change from v2 0.870566; added ranking baselines did not exceed EASE. Negative results and extra follow-up budgets are retained.
