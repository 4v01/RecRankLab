# V3 frozen study: rating accuracy and full-catalog ranking

The study/configuration is committed before running ML-1M v3 final tests. Prior v1/v2 test results were known and prompted this follow-up; it is not an untouched benchmark. No v3 test result is supplied to training, epoch selection, tuning or model proposals.

The original per-user temporal split, official GroupLens archive, processed data hashes, supported catalog, cold handling, all-history masking, K=10/20 and metric definitions remain unchanged. Raw data and checkpoints remain ignored. Main rating RMSE/MAE stay **unclipped**, on all observed held-out ratings. Explicitly labeled clipped auxiliary RMSE/MAE provide a diagnostic of a common external convention; they are never selected on and do not replace the primary estimand.

## Separate objectives

V2 selected every model by NDCG@10. V3 rating models (BiasMF, item AutoRec, residual item AutoRec) select by minimum **validation RMSE**. Five ranking models (ItemKNN, RP3beta, SLIM, MultiVAE, EASE) select by maximum **validation NDCG@10**. Report these in separate primary tables; their cross-task auxiliary scores do not have equal selection objectives. Each receives four declared parameter candidates, seeds17/29/43. Exact settings: configs/v3/study.yaml and model YAMLs.

## Validation stopping and final refit

Iterative models evaluate the validation objective every five epochs, and at the final allowed epoch if necessary. Improvement must exceed 1e-5; retain the first best checkpoint under that rule. Stop after three nonimproving checks. Max60 epochs for MF/VAE, max80 for AutoRec. Each trial records every checked epoch, metrics, best epoch, stopped epoch and all training losses. Select the best candidate using its retained validation score. Persist the selected parameter configuration and epoch before accessing the test evaluator. Reset the seed, initialize a new model, refit train+validation for exactly that epoch count, then test once. No training checkpoint is evaluated on test to decide epochs. Existing completed signatures are reused and incompatible overwrites are rejected.

Parameter candidate counts are equal, but wall time and validation-checkpoint counts are not. Both are reported. Early stopping itself exposes more validation scores than a single analytical fit; do not describe four candidates as equal statistical search effort or equal compute. Deterministic classical-model seeds are reproducibility checks, not independent stochastic fits.

## Implementations

- ItemKNN: binary fit positives, cosine co-occurrence similarity with additive shrinkage. Zero diagonal; stable top-K source neighbors per target column.
- RP3beta: positive-only normalized item→user and user→item transition matrices, elementwise power alpha, target item degree^-beta. Zero diagonal, top-K source neighbors per target column, then row-L1 normalization. Scores multiply binary user histories by this item matrix. This precise sparsification/normalization variant is declared, rather than claiming bitwise identity with every RP3beta implementation.
- SLIM: one nonnegative ElasticNet regression per target item, no intercept, self column zeroed; scikit-learn's objective is SSE/(2*n_users)+alpha*l1_ratio*L1+alpha*(1-l1_ratio)/2*L2. Two workers own separate CSC matrices; they cannot alter another target's features. Max1000 coordinate-descent iterations, tol1e-3. A target's dual gap exceeding sklearn's tolerance makes the trial fail explicitly. Failed trials count against the four-candidate budget; final failure is not silently replaced by a test-favored model. Coefficients and diagnostics are retained.
- MultiVAE: L2-normalized binary positive inputs; input dropout0.5; tanh encoder to mean/logvariance (latent128); train by reparameterization and multinomial negative log-likelihood plus annealed Gaussian KL. Decode the posterior mean with dropout disabled at evaluation. KL reaches the candidate cap over20 fit epochs, based solely on fit batch counts. This dataset-sized annealing schedule differs from the original ML-20M absolute-step schedule. Original notebook's Recall denominator is not imported; our evaluator remains authoritative.
- BiasMF: original observed-rating MSE model, stronger predeclared regularization and validation stopping.
- AutoRec: original observed-only item reconstruction, two hidden sizes and two weight decays, input corruption0.2; RMSE stopping. Residual variant first fits a regularized mean+user/item bias using only fit observations (10 alternating bias updates, shrinkage10), reconstructs observed residuals and adds the baseline at scoring. Its observation mask is preserved even when a residual is zero. Plain/residual variants have matching grids and stopping rules but are tuned separately, so this is a pipeline-with-tuning comparison.
- EASE: rerun the four ridge candidates as an in-study full-catalog reference.

## Uncertainty, provenance and limits

Export anonymous ranking rows and, for rating models, user-level count/SSE/SAE sufficient statistics, never ratings or prediction pairs. Report means and sample SD over seeds. Ranking differences bootstrap users after averaging seed differences. Rating differences bootstrap the same users jointly for both models, compute each seed's sqrt(total SSE/total count), then average the seed RMSE differences. Use2000 resamples, seed2026. These intervals condition on the reused split and fitted models, not new-data or search uncertainty.

All runs record clean code SHA, config and selected config, data/split fingerprints, seed, environment, runtime and peak process-lifetime RSS. V3 reports and manifest are separate from immutable v2 reports. New external rating references and their caveats live under research/ with a separate source manifest; no external reported RMSE is a local replication or a target that justifies test leakage, clipping the main metric, changing splits or dropping difficult ratings.
