# Known limitations

- The old test results were read before formulating v2 hypotheses. New validation-only tuning does not erase benchmark reuse or researcher-level selection. An untouched later/global-forward dataset is still needed for independent generalization claims.
- Per-user holdout is not a global time cutoff. Thresholded ratings are not real implicit clicks, missing items are not confirmed negatives, and support restrictions remove some cold targets.
- Core and improved systems have different tuning/training budgets, explicitly reported. Improvement variants change several factors together; they do not identify a single causal mechanism.
- Four candidates remain a small search. Fixed epochs do not establish convergence; poor explicit-model ranking need not imply a bug or inferiority of every rating objective.
- The self-optimizer is a bounded deterministic ridge search with validation memory. It does not reproduce CogAlpha/XALPHA, implement autonomous code research, or establish that adaptive search beats grid search. Its four-fit ablation tests this limited claim directly.
- EASE is deterministic and quadratic in catalog size; its three seeds are duplicate numerical checks. Zero seed variance is not zero generalization uncertainty. Gradient parameter counts exclude its dense learned coefficient matrix.
- User bootstrap conditions on a single reused split and does not quantify new-dataset uncertainty or account for multiple exploratory comparisons.
- Local checkpoint hashes are recorded, but checkpoints/data are deliberately not distributed. Reproduction requires the official dataset download and full Git history.
- Uploaded v1 evidence is preserved separately in reports/legacy_v1. Its original Git history was absent from the ZIP and its provenance cannot be independently established here. New headline tables include only new locally executed v2 runs.
- Exact numerical agreement is tied to the locked CPU environment, not arbitrary CPUs, GPUs or dependency versions. RSS is process-lifetime high-water memory.
- Optional LLM reranking, a global-forward sensitivity experiment and new-dataset confirmation are not implemented in this study.

## Historical descriptive metadata errata

The immutable v2 evidence was generated at source `ab7798096d563205153f632d1029c59559cd10aa`. Its Popularity hypothesis descriptor inherits a budget of 2 from the base config, although the actual validation trial list, study plan and reported fit count correctly show 1. Its BLAS environment snapshot precedes per-fit thread setup, so the first seed may list startup thread counts; `seed_all` actually sets both BLAS and PyTorch to the configured two threads before each fit. Future runner snapshots and Popularity descriptors are corrected in the later reporting commit. Raw records are preserved, not rewritten. Neither correction changes scores, selection or the evaluated population.
