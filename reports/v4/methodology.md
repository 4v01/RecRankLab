# V4 frozen before new final evaluations

This is exploratory benchmark reuse after v1–v3. New ML-10M is a separate benchmark, not a causal sample-size intervention. Keep ML-1M as the continuity anchor; no old result/evidence is overwritten. Do not promise improved metrics or change protocols to make them increase.

## Protocol and data

Use complete official ML-1M and ML-10M, including half-star ML-10M ratings; no user/item subset. Same per-user temporal floor(.8n)/floor(.9n), original item-ID and source-row tie ordering, ≥4 positives, fit-supported full catalog and masking of every previously rated item. ML-10M is not globally forward-in-time. Fit train only for validation, clean seed reset and train+validation refit at validation-selected epochs, final test once. Unclipped observed-rating RMSE/MAE/R²; auxiliary clipping uses each dataset's actual lower bound (1 vs0.5). On different datasets, absolute RMSE/R²/NDCG are not directly evidence of improvement due to size alone.

Full-catalog evaluation now materializes candidates lazily, batches scores, and finds exact stable top-K via partition plus complete boundary-tie handling. Tests compare masks, scores, stable ranking and metrics against the original small-data path. Sparse EASE uses the same closed-form objective, sparse histories and requested-user scoring, avoiding dense full-user score caches. Sparse reduction may differ in floating-point rounding; it is not a change in candidate policy or relevance labels.

## Frozen experiments and budgets

configs/v4/study.yaml is authoritative: each dataset has BiasBaseline and SparseEASE (one seed17 each; deterministic), SGD MF, MF-DCN and HistoryTower (seeds17/29/43 each). Two parameter candidates per base model: 11 final base fits per dataset, 22 total. Two hybrid variants × three seeds = six evaluations per dataset; total34 final base/hybrid records across both datasets. A hybrid reuses final base checkpoints and does not pretend to add independent fits.

SGD MF: per-rating sequential SGD, observed MSE, fit-only ten-update bias warm start, original-factor simultaneous update, decay0.98/epoch, dimensions64/128, learning rate0.005 and factor/bias regularization0.02. Max60 epochs, validation every2, patience5, min_delta1e-5. Native kernel is original C++17, -O3, no fast-math, FP contraction disabled, single thread; source hash/compiler/flags recorded. Loss history is pre-update online MSE, not the full regularized end-of-epoch objective.

MF-DCN: fit-only bias warm start + MF dot product + two original vector cross layers and parallel64→32 ReLU MLP; linear output, dropout0.1, dimensions16/32, Adam0.001, observed embedding L2=0.001 and global weight decay1e-5. Max35 epochs, every2 checks, patience4. This is a deliberately small rating adaptation; only IDs, no claimed production features. Rating-only methods omit auxiliary full-catalog ranking in both datasets; they share the complete observed-rating evaluator. DCN is not trained to optimize NDCG.

HistoryTower: recent50 fit-positive items, mean pooling plus user-ID embedding, independent two-layer towers, normalized64-dimensional vectors, temperatures0.1/0.2, Adam0.001. Each epoch samples one positive uniformly per supported user, balanced by users rather than rating events. Current positive is excluded from its own history; all other known fit positives in batch columns are masked as negatives. Max80 epochs, every5 checks, patience3. No future-positive blacklisting. It takes only architecture inspiration from X's history-conditioned retrieval, not Phoenix code/weights/loss fidelity; no Transformer or multimodal/multi-action features are claimed.

BiasBaseline: 10 alternating observed bias updates, shrink10/25. SparseEASE: ridge300/1000. Analytical methods have no epoch search. Search counts match by candidates but compute, validation exposure, architecture and user weighting differ.

## Frozen combinations

Rating blend: w·SGD MF+(1−w)·MF-DCN. Ranking blend: w·SparseEASE+(1−w)·HistoryTower after per-user z-score over the fixed fit-supported catalog, including seen entries in normalization but masking them for recommendation. A requested candidate's score does not depend on which other candidates were requested. Five weights0/.25/.5/.75/1 are declared, choose by validation RMSE or NDCG@10; ties choose first. Endpoint choices and failures remain results, not suppressed.

For each component, reproduce its selected model on train only at its already selected epoch, evaluate combinations on validation, persist weights/records, then reuse train+validation final component checkpoints for one hybrid test. Validation component caches are keyed to selected config, original code commit and data and checked by content hash. The hybrid stage has additional selection and retraining budget and previously exposed component tests: no untouched-confirmation or equal-compute claim. Every final record binds source SHA/config/data/seed and dependency signatures/checkpoint hashes. Repeating a signature is reused; changing signatures cannot overwrite final tests.

## Diagnostics and uncertainty

Diagnose data density, cold fractions, simple fit-only mean/bias validation baseline, support cohorts, full-catalog difficulty and historical checkpoint boundaries. These describe plausible bottlenecks, not experimental proof that any single cause explains all errors. Larger data may be sparser. Report per-seed outcomes and negative hypotheses. Conditional user paired bootstrap resamples anonymous per-user statistics, with2000 resamples seed2026; this omits new-data, search and researcher-direction uncertainty. Do not pool cross-dataset users or compare deterministic repeats as independent seeds. No paid service, API or GPU is needed. Raw data/prediction pairs/checkpoints remain untracked.

V4 checkpoints store only fitted support/mean/parameters and necessary sparse histories, rather than repeated full raw rating frames. Round-trip tests exercise compact and original formats; original checkpoints still load unchanged. This keeps all full ML-10M models and blend dependencies within the available disk without subsampling data.
