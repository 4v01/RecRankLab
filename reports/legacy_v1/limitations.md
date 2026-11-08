# Known limitations

1. This is a small, fixed-budget CPU benchmark. Two learning rates and 20 epochs do not establish convergence or optimal performance. A more complex model can legitimately lose to Popularity. No result was required to beat a baseline.
2. Per-user temporal holdout is not globally chronological. Cross-user future knowledge and arbitrary ordering of equal timestamps remain possible. There is no global-forward split sensitivity experiment.
3. Ratings >=4 are a proxy for relevance. Unknown items are not verified dislikes. Train-only sampling can select a future positive; filtering it with future labels would leak information.
4. Unsupported items are excluded from ranking, so this is a warm-item benchmark. Explicit metrics include observed cold cases via fit-mean fallback. No side-information cold-start model exists.
5. U-AutoRec is a documented implementation variant, not a reproduction of the original paper's exact hyperparameters or strongest item-based results. There is no verified comparison to the unrelated historical RMSE 0.13 claim.
6. The model comparison does not isolate loss functions: architectures, initialization, regularization formulation and optimization differ. The BPR sampling ablation controls the architecture and tuning budget but tunes each policy separately.
7. Three seeds give limited evidence about optimization variance. User-bootstrap intervals condition on one dataset and split; dependent user-item interactions and hyperparameter-search uncertainty are not captured. No multiple-comparison correction is applied; intervals are descriptive.
8. The implementation stores dense score/input matrices for simplicity. It fits ML-1M on CPU but is not designed for million-item catalogs. Checkpoints include fit interactions locally and must not be published as generic artifacts. Raw data, derived split files and checkpoints stay gitignored.
9. Peak RSS is Linux process-lifetime high-water mark including previous trials/seeds, not isolated model peak memory. CPU timings depend on host load; no GPU experiment is claimed.
10. CI configuration includes offline synthetic correctness tests and an official-download ML-100K smoke job. The smoke job depends on GroupLens and package-index availability. Local passage is not evidence of a completed hosted GitHub Actions run.
11. Dependency versions are locked for the tested Python 3.12 Linux CPU environment; package hashes and hardware-independent bitwise reproducibility are not guaranteed. No online experiment or external independent audit has been performed.
12. Optional LLM re-ranking, global-forward sensitivity, wider searches, AutoRec unmasked-loss diagnostic and graph-depth sweep are not implemented. The mandatory ablation is BPR negative-sampling policy.
