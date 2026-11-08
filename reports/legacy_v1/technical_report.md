# RecRankLab — technical experiment report

Generated from 18 completed ML-1M runs. Full planned suite complete: **True**.

This report describes this fixed-budget implementation, not a reproduction of published benchmark scores.

## Main results

Mean ± sample standard deviation over seeds 17, 29, 43. Higher ranking metrics are better.

| Model | Recall@10 | NDCG@10 | Recall@20 | NDCG@20 | RMSE | Final fit seconds |
|---|---:|---:|---:|---:|---:|---:|
| autorec | 0.02458 ± 0.00556 | 0.02703 ± 0.00827 | 0.04205 ± 0.00957 | 0.03229 ± 0.00916 | 0.95479 ± 0.02577 | 10.24864 ± 0.33987 |
| biasmf | 0.01526 ± 0.00050 | 0.01812 ± 0.00035 | 0.02684 ± 0.00029 | 0.02153 ± 0.00034 | 0.95914 ± 0.00320 | 16.84294 ± 0.41226 |
| bpr_mf | 0.08054 ± 0.00136 | 0.08500 ± 0.00154 | 0.13404 ± 0.00174 | 0.10043 ± 0.00157 | — | 11.03196 ± 0.51837 |
| lightgcn | 0.08613 ± 0.00118 | 0.08845 ± 0.00041 | 0.14516 ± 0.00158 | 0.10569 ± 0.00050 | — | 72.13931 ± 1.49865 |
| popularity | 0.04934 ± 0.00000 | 0.05842 ± 0.00000 | 0.08626 ± 0.00000 | 0.06812 ± 0.00000 | — | 0.03017 ± 0.00109 |

## Shared evaluation population

All runs share 5820 eligible users and 3683 training-supported items. Excluded cold positive targets: 9 of 53025. Catalog items without fit support: 23.

## Paired comparisons

Bootstrap unit: user. Average each user’s paired metric differences over the three seeds, then resample users 2,000 times (bootstrap seed 2026). These intervals condition on the split and do not represent uncertainty over future datasets or hyperparameter searches.

| Comparison | NDCG@10 difference | 95% user bootstrap interval |
|---|---:|---:|
| biasmf - popularity | -0.040299 | [-0.043274, -0.037271] |
| autorec - biasmf | 0.008909 | [0.007200, 0.010624] |
| bpr_mf - biasmf | 0.066878 | [0.063673, 0.070265] |
| lightgcn - bpr_mf | 0.003454 | [0.002075, 0.004892] |
| bpr_positives_only - bpr_mf | -0.000757 | [-0.002169, 0.000662] |

## Negative-sampling ablation

The BPR positives-only variant may sample previously observed low ratings as negatives; the headline BPR excludes all prior ratings. Both variants use the same two learning rates, fixed epochs and train/validation/test protocol, and tune independently on validation. This estimates policy performance under equal tuning budget, not a single fixed-hyperparameter causal contrast. See ablations.csv and paired_differences.csv.

## Research interpretation

RQ1: Compare the observed RMSE and NDCG in the table and rating_vs_ranking.png; two models and three seeds cannot establish a general monotonic relationship. RMSE and ranking use different target definitions and populations.
RQ2: BPR vs BiasMF/AutoRec differences jointly reflect objective, architecture and optimization. They do not isolate the loss function alone.
RQ3: LightGCN vs BPR uses identical embedding width, negative policy, candidate universe and tuning budget. Graph propagation changes compute cost. See the paired interval above before claiming any benefit.

## Reproducibility and audit

See methodology.md for the frozen protocol, limitations.md for known deficiencies, experiment_manifest.json for artifact hashes and all run-level provenance, and evidence/ for validation trials, selected configurations, loss curves, environment, timestamps and anonymized per-user metrics. Raw ratings, split row data and model checkpoints are not redistributed. A clean commit is required by the runner; final outputs cannot be silently overwritten.

## Runtime

final_fit_seconds excludes tuning, checkpoint serialization and evaluation. total_seconds includes them; evaluation_seconds includes full-catalog scoring and observed-rating metrics. RSS is a process-lifetime Linux high-water mark, not an isolated per-model allocation. Hardware and versions are captured per run.

## Scope

LLM re-ranking was not implemented. No historical AutoRec RMSE claim is reused. No model is required to beat Popularity. Local tests and smoke validation are documented separately in validation.md.
