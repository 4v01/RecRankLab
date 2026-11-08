# Findings and diagnostic interpretation

These observations were written after the predeclared runs. They did not trigger model, split, candidate, epoch or test changes. Numeric values below are generated from saved evidence.

BPR-MF mean NDCG@10 is 0.084996; Popularity is 0.058417; their absolute difference is 0.026579. This is an implementation-specific fixed-budget observation, not a claim of general superiority.

AutoRec mean RMSE is 0.954793 and NDCG@10 is 0.027026 with seed SD 0.008273. Selection differs across seeds; see validation_trials.csv. The model's sensitivity to initialization and the small search budget limits stable conclusions.

LightGCN minus BPR mean NDCG@10 is 0.003454. LightGCN final fit takes 72.14s on average versus 11.03s for BPR. See paired_differences.csv for the user-level interval, not only the ranking of means.

## Rating error is not the selection objective

For AutoRec seed 17, the two validation trials provide a concrete objective-mismatch example:
- Learning rate 0.001: validation RMSE 0.971799; NDCG@10 0.016755.
- Learning rate 0.003: validation RMSE 0.935937; NDCG@10 0.016097.

The chosen trial maximizes validation NDCG@10 even if the other trial has a lower RMSE. This is a local counterexample to equating the two metrics, not an estimate of a universal relationship.

## Observed weaknesses

BiasMF and AutoRec both rank below Popularity under the tested setup. The observed-rating objective does not penalize predictions on missing items. BiasMF’s final training objective is substantially lower than held-out squared error; overfitting and temporal distribution shift are plausible contributors. This does not establish which mechanism dominates. No additional test-guided tuning was performed.

Training curves are included for all four trainable headline systems. Declining training loss alone does not establish converged validation ranking. These model-specific objectives have different scales and should not be compared numerically with one another.

## Follow-up experiments, not completed claims

A future preregistered study could broaden regularization and epoch searches using validation only, compare U-AutoRec with I-AutoRec, and add a globally forward split. These follow-ups must be reported separately; the existing test set should not become an informal tuning set. Optional LLM reranking remains deferred.
