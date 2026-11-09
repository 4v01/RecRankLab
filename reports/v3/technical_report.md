# V3 actual-results report

Primary tables use new ML-1M run evidence; explicitly separated v2 references remain immutable.
All declared phases complete: **True**; final runs: 30. Primary phase: 24 runs, low-decay follow-up: 3, SVD++ reference follow-up: 3. The follow-ups were separately frozen; they are not retroactively part of the original eight-experiment registration.

## Rating prediction: validation RMSE selection

| Model / phase | Unclipped RMSE | R² | MAE | Clipped RMSE (auxiliary) | Epoch | Total seconds |
|---|---:|---:|---:|---:|---:|---:|
| autorec_lowreg_v3 (low_decay_followup) | 0.879918 ± 0.000831 | 0.410741 ± 0.001112 | 0.691266 | 0.879458 | 41.7 | 273.7 |
| autorec_residual_v3 (primary) | 0.876138 ± 0.000844 | 0.415793 ± 0.001125 | 0.687525 | 0.875655 | 20.0 | 215.8 |
| autorec_rmse_v3 (primary) | 0.869547 ± 0.000891 | 0.424549 ± 0.001179 | 0.684992 | 0.869359 | 76.7 | 325.7 |
| biasmf_rmse_v3 (primary) | 0.877868 ± 0.000588 | 0.413483 ± 0.000785 | 0.688846 | 0.877620 | 5.0 | 88.2 |
| svdpp_rmse_v3 (svdpp_reference_followup) | 0.873099 ± 0.000480 | 0.419839 ± 0.000637 | 0.686523 | 0.872695 | 29.0 | 203.6 |

Historical v2 item AutoRec RMSE: 0.870566; it was NDCG-selected with a different grid and fixed epochs. This descriptive reference does not isolate early stopping or objective selection. V2 rating/model evidence remains in ../evidence/.

## Ranking: validation NDCG@10 selection

| Model | NDCG@10 | Recall@10 | NDCG@20 | Recall@20 | Total seconds |
|---|---:|---:|---:|---:|---:|
| ease_v3 | 0.101701 ± 0.000000 | 0.102477 | 0.121762 | 0.170378 | 31.8 |
| itemknn_v3 | 0.087892 ± 0.000000 | 0.084290 | 0.104166 | 0.141573 | 26.9 |
| multivae_v3 | 0.088144 ± 0.000507 | 0.092654 | 0.109218 | 0.158741 | 481.8 |
| rp3beta_v3 | 0.088343 ± 0.000000 | 0.087113 | 0.105958 | 0.147609 | 28.1 |
| slim_v3 | 0.098923 ± 0.000000 | 0.101380 | 0.117944 | 0.165882 | 145.8 |

## Paired uncertainty

| Comparison | Metric | Mean difference | 95% user bootstrap |
|---|---|---:|---:|
| itemknn_v3 - ease_v3 | ndcg@10 | -0.013809 | [-0.016528, -0.011174] |
| rp3beta_v3 - ease_v3 | ndcg@10 | -0.013359 | [-0.016653, -0.010060] |
| slim_v3 - ease_v3 | ndcg@10 | -0.002778 | [-0.004844, -0.000575] |
| multivae_v3 - ease_v3 | ndcg@10 | -0.013557 | [-0.016120, -0.010939] |
| autorec_residual_v3 - autorec_rmse_v3 | rmse | 0.006591 | [0.004334, 0.009207] |
| autorec_rmse_v3 - biasmf_rmse_v3 | rmse | -0.008321 | [-0.011007, -0.005841] |
| autorec_lowreg_v3 - autorec_rmse_v3 | rmse | 0.010371 | [0.009120, 0.011634] |
| svdpp_rmse_v3 - biasmf_rmse_v3 | rmse | -0.004770 | [-0.006255, -0.003275] |
| svdpp_rmse_v3 - autorec_rmse_v3 | rmse | 0.003551 | [0.000852, 0.006352] |

## Audit and limitations

All models share 5820 ranking users, 3683 supported items and 9 cold-positive exclusions. Rating models use 102759 observed test ratings. Data fingerprints and cohort hashes are checked across runs.
Hardware: INTEL(R) XEON(R) PLATINUM 8573C, 3 visible logical CPUs, configured two PyTorch/BLAS threads, CPU only. Python 3.12.14, Torch 2.15.0+cpu. RSS is process-lifetime high-water memory, not isolated per-model allocation.
Every experiment has four parameter trials; ordinary AutoRec has four original decay candidates plus four separately frozen low-decay candidates (eight total settings in that family), while residual AutoRec is a distinct four-candidate variant. SVD++ is an additional four-candidate model with every-epoch validation, versus every five epochs for the primary iterative models. These are disclosed additional search/compute budgets. Iterative models additionally search validation checkpoints. Counts and full histories are exported. Compute and validation exposure differ; this is not equal-time or equal-search-effort causal attribution.
SLIM records convergence diagnostics; failures count against budget and are not silently scored. Deterministic classical model seeds are reproducibility repeats. MultiVAE evaluation uses posterior means. Original unobserved feedback, support, masking and Recall/NDCG definitions are unchanged.
R² is 1-SSE/SST with SST centered on the observed test mean; it can be negative and is not squared correlation or a causal/mechanistic explanation. Previously completed raw JSON records are immutable: missing R² is derived from their saved RMSE/count and the shared observed test SST, with origin recorded in the manifest. Unclipped RMSE remains primary. Clipped metrics are diagnostic only and do not drive selection. External lower RMSE figures use different protocols, and one inspected AutoRec implementation inputs test ratings to its test reconstruction; see research/README.md for exact pinned evidence.
This is exploratory benchmark reuse: v1/v2 and earlier-phase results were visible when later phases were declared. Each phase freezes its parameter candidates before its own final tests; test observations never enter fitting, negative sampling, epoch selection or parameter selection. The low-decay hypothesis was frozen from validation before inspecting the new AutoRec test scores. Conditional user bootstrap omits new-data, search, researcher-direction and multiple-comparison uncertainty. See methodology.md, experiment_manifest.json and evidence/; data/checkpoints are not redistributed.
