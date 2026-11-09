# Scoring dtype maintenance

SparseEASE stores float32 coefficients and float64 CSR histories. Implicit promotion copied the entire item matrix for every user batch. The maintenance fix retains one float64 scoring view and invalidates it after refitting or state loading. The original parameter tensor, serialization and scoring equation remain unchanged. A test checks exact values through repeated scoring, refitting and state reload; all69 tests passed locally.

A synthetic warm-path microprofile (128×4096 CSR,4096×4096 coefficients, seed17, five repeated calls) took0.303171s with repeated promotion and0.127620s with the retained view, ratio2.376. Outputs are elementwise identical. This is not an end-to-end training speedup or recommendation-accuracy result. Timings depend on hardware, sparsity and caching; the full synthetic-data fingerprint, packages, clean source SHA and configuration are in ease_dtype_microprofile.json.

The tradeoff is an additional float64 view per active model. Item-matrix storage remainsO(items²), so this does not remove the large-catalog memory limit. The view is not serialized. All delivered recommendation metrics remain bound to their original numerical experiment source commits and have not been rerun or selected after this maintenance change.
