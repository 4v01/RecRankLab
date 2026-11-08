# RecRankLab v2 — actual experiment report

Completed 33 new ML-1M final fits. Frozen study complete: **True**.

Historical evidence in legacy_v1 is excluded. Test scores from v1 were known before designing v2; this is a reused benchmark, not an untouched confirmation set.

| Model | Seeds | Validation fits/seed | Recall@10 | NDCG@10 | Recall@20 | NDCG@20 | RMSE | Final fit s | Total s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| autorec | 3 | 2 | 0.025346 ± 0.006417 | 0.027863 ± 0.007931 | 0.043817 ± 0.008130 | 0.033435 ± 0.007933 | 0.942738 ± 0.014198 | 7.430484 ± 1.534317 | 31.179096 ± 1.596816 |
| autorec_item | 3 | 4 | 0.039652 ± 0.001020 | 0.046278 ± 0.001501 | 0.067267 ± 0.000820 | 0.053741 ± 0.001228 | 0.870566 ± 0.000958 | 33.690380 ± 7.099589 | 200.423174 ± 27.677863 |
| biasmf | 3 | 2 | 0.015260 ± 0.000502 | 0.018118 ± 0.000353 | 0.026836 ± 0.000286 | 0.021526 ± 0.000341 | 0.959141 ± 0.003200 | 11.812672 ± 0.915018 | 43.693746 ± 3.212323 |
| bpr_mf | 3 | 2 | 0.080537 ± 0.001363 | 0.084996 ± 0.001543 | 0.134041 ± 0.001740 | 0.100426 ± 0.001572 | — | 7.915297 ± 0.048552 | 31.000595 ± 1.754513 |
| bpr_optimized | 3 | 4 | 0.090632 ± 0.001570 | 0.086720 ± 0.001162 | 0.150599 ± 0.001093 | 0.105097 ± 0.001039 | — | 72.042929 ± 12.435288 | 336.260552 ± 25.302474 |
| bpr_positives_only | 3 | 2 | 0.079369 ± 0.000637 | 0.084239 ± 0.000125 | 0.132711 ± 0.002079 | 0.099917 ± 0.000780 | — | 10.601720 ± 0.205808 | 37.554403 ± 0.944740 |
| ease | 3 | 4 | 0.102477 ± 0.000000 | 0.101701 ± 0.000000 | 0.170378 ± 0.000000 | 0.121762 ± 0.000000 | — | 3.059116 ± 0.429891 | 33.998763 ± 1.340150 |
| ease_selfopt | 3 | 4 | 0.102477 ± 0.000000 | 0.101701 ± 0.000000 | 0.170378 ± 0.000000 | 0.121762 ± 0.000000 | — | 2.723195 ± 0.092413 | 31.681974 ± 1.639477 |
| lightgcn | 3 | 2 | 0.086129 ± 0.001176 | 0.088450 ± 0.000410 | 0.145153 ± 0.001577 | 0.105684 ± 0.000490 | — | 40.969741 ± 2.677182 | 118.818985 ± 2.659228 |
| lightgcn_optimized | 3 | 4 | 0.089878 ± 0.002677 | 0.091230 ± 0.001111 | 0.151375 ± 0.005269 | 0.109009 ± 0.002121 | — | 88.209720 ± 5.014666 | 388.995146 ± 22.081236 |
| popularity | 3 | 1 | 0.049337 ± 0.000000 | 0.058417 ± 0.000000 | 0.086258 ± 0.000000 | 0.068119 ± 0.000000 | — | 0.026218 ± 0.003888 | 6.418580 ± 0.795980 |

Every model shares 5820 eligible users, 3683 supported items, and 9 excluded cold positives out of 53025 held-out positives. Full catalog; prior train+validation interactions masked.

## Paired user bootstrap

Seed-averaged user differences; 2,000 user resamples, seed 2026; conditional percentile intervals. Not corrected for benchmark reuse, multiple comparisons or search uncertainty.

| Comparison | NDCG@10 difference | 95% interval |
|---|---:|---:|
| bpr_mf - popularity | 0.026579 | [0.023381, 0.029969] |
| lightgcn - bpr_mf | 0.003454 | [0.002075, 0.004892] |
| autorec_item - autorec | 0.018415 | [0.016450, 0.020452] |
| bpr_optimized - bpr_mf | 0.001724 | [-0.000632, 0.004092] |
| lightgcn_optimized - lightgcn | 0.002779 | [0.001452, 0.004151] |
| ease - lightgcn_optimized | 0.010472 | [0.007601, 0.013175] |
| ease_selfopt - ease | 0.000000 | [0.000000, 0.000000] |
| bpr_positives_only - bpr_mf | -0.000757 | [-0.002169, 0.000662] |

## Hardware and software

Runs used INTEL(R) XEON(R) PLATINUM 8573C; 3 visible logical CPUs, two configured PyTorch/BLAS threads, CPU only. The container was observed with a two-CPU quota and 8 GiB memory limit. Python 3.12.14, PyTorch 2.15.0+cpu, NumPy 2.5.4, pandas 2.2.3, SciPy 1.18.1. These are shared-cloud wall times, not isolated hardware performance measurements.

EASE stores a 3706×3706 float32 coefficient matrix (52.39 MiB; 13,730,730 off-diagonal entries), in addition to inputs and cached scores. Its zero gradient-parameter count does not mean zero learned storage.

## Interpretation and limits

Original-budget core baselines have two validation trials and 20 epochs. Improvement variants have four trials and longer training; their differences combine architecture, optimizer and budget. Do not describe them as equal-compute causal comparisons. EASE has four analytical fits; its learned dense matrix is not counted as gradient-trained parameters and scales quadratically in item count.

The negative-policy contrast and equal-four-fit EASE grid/memory-search comparison are reported in ablations.csv. The self-optimizer uses validation hypotheses and elite memory; a lower or tied test result remains in the report. It is not a CogAlpha/XALPHA reproduction or LLM system. Deterministic EASE seeds are duplicate checks, not independent uncertainty draws.

The explicit RMSE and ranking scatter compares different targets/populations and is descriptive. No universal relation or universal superiority follows. All unsupported targets and users follow the shared evaluator.

See methodology.md for frozen rules, limitations.md for known issues, validation_trials.csv for all successful/failed search candidates, per_seed_results.csv for hardware/resource metrics, and experiment_manifest.json/evidence for commit/config/data/seed provenance. Checkpoints, splits and raw data are not distributed. RSS includes earlier trials in the process; final fit excludes tuning, total time includes it. Optional LLM reranking remains deferred.
