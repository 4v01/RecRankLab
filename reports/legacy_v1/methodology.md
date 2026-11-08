# Methodology — frozen before headline test runs

## Data and estimand

Official GroupLens MovieLens 1M is the main benchmark; 100K is a separate smoke dataset. Download ZIP archives directly from https://files.grouplens.org/datasets/movielens/. SHA-256 records both downloaded archive and rating member. Parse all recorded ratings; reject duplicate user–item pairs. Structural user/item IDs are mapped from sorted original IDs; this declares an ID universe, but no held-out rating statistic enters fit. Unsupported IDs are excluded from ranking and counted. Movie metadata is unused.

Per user, stable sort by (timestamp, original item ID, original row ID); train ends at floor(0.8n), validation at floor(0.9n), ensuring at least one observation in each partition. Save integer split indices, mappings, counts and hashes locally. This is user-relative temporal holdout, **not a single global cutoff**. Equal-timestamp ordering is arbitrary but deterministic. Items supported by other users can come from a later calendar time than this user's target.

## Views and candidates

Explicit models train on all fit ratings. Positives for implicit models and ranking relevance are ratings >=4. Candidate catalog is every item with at least one observed rating in the current fit partition, regardless of rating value, minus all the user's previously recorded ratings. This catalog is identical for every model. Validation history is train; final test history is train+validation. No sampled-negative headline metrics. Higher score is always better. Equal scores break by ascending original item ID. Only users with at least one supported, unconsumed held-out positive and some fit history enter ranking metrics. Users without positives, unsupported targets, supported catalog size and eligible-user hash are reported. All rating models evaluate all observed held-out ratings, including low ratings. Cold user/item rating predictions fall back to the fit global mean; predictions are **not clipped** to 1–5.

Recall@K = hits / number of eligible positives. NDCG uses binary relevance, log2(rank+1) discount and ideal DCG for min(K, number of eligible positives). Macro average over the same eligible users. Coverage@K is unique recommended items / supported catalog size. K=10,20. If fewer than K candidates exist, return all available candidates, never masked fillers.

## Models

- Popularity: positive fit counts only, no parameters.
- BiasMF: global fit mean + user/item bias + 32-dimensional inner product; Adam, observed MSE and sampled-parameter L2 penalty.
- User-based AutoRec: training-only zero-filled rating vector; 64-unit sigmoid encoder and linear decoder; squared error indexed only by observation mask, Adam with L2 weight decay. The observation mask is used in the loss, not concatenated as an extra feature. No input normalization using future statistics, no denoising and no hidden-layer dropout. Cold item outputs use the same explicit mean fallback as BiasMF.
- BPR-MF: 32-dimensional user/item embeddings, normal initialization with SD=0.1, softplus negative score difference, L2 penalty on sampled initial embeddings. Each epoch visits all positive fit pairs once in shuffled order and samples one negative per pair. Uniform negatives come only from the fit-supported catalog and exclude every recorded fit interaction.
- LightGCN: symmetric D^(-1/2) A D^(-1/2) bipartite graph of positive fit pairs, no self loops, features or nonlinear propagation, uniform average of embedding layers 0,1,2. Same BPR loss, sampling and initial embedding regularization as BPR-MF. Graph is reconstructed independently for final train+validation fit.

## Negative sampling and future information

The sampler takes only the fit partition. It never queries validation/test interactions to blacklist future positives. Such blacklisting would reveal future consumption. Thus a supported item that is a future positive can still be sampled as unknown; false negatives are possible and explicitly accepted. Future-only items with no fit support cannot be sampled. This resolves the brief's conflicting “no future positives” and “no held-out information in negative sampling” requirements in favor of a leakage-free, train-only sampler. Tests verify deterministic sampling, no fit-history samples, no future-only items and invariance to changes in future labels.

## Predeclared tuning budget and final test

Seeds: 17,29,43. For each seed, each trainable model receives exactly two learning-rate candidates, each trained for exactly 20 epochs, then evaluated on validation. BiasMF: {0.003,0.01}; AutoRec: {0.001,0.003}; BPR-MF and LightGCN: {0.01,0.03}. Regularization=0.0001, embedding dimension=32, hidden size=64, graph depth=2. Batches: rating MF 8192, ranking 32768, AutoRec 128 users. Two PyTorch CPU threads, deterministic algorithms. Seeds are reset before each candidate so initial states are paired within a model. Select maximum validation NDCG@10, tie to first candidate; persist selection before final fit. No validation early stopping or test-dependent epoch choice. Reinitialize, train on train+validation for the selected fixed epochs and evaluate test once per seed. Hyperparameters are selected separately per seed. Popularity gets a single nonparametric fit.

Equal candidate count and data passes provide a transparent **limited tuning budget**, not equal wall time or proof of convergence. Hyperparameter ranges are intentionally small. Runtime and parameter counts expose differences in computational effort. Re-running a completed signature reuses it; a different signature cannot silently overwrite final outputs. A clean Git commit is mandatory except for explicitly marked development runs.

## Ablation and uncertainty

Predeclared BPR negative-policy ablation: exclude fit positives only, allowing observed low ratings as negatives, versus exclude all recorded history. Same tuning budget and final evaluation; neither uses future records. Independently selected hyperparameters mean this is a policy-with-tuning contrast. No intentionally broken model is included in headline results.

Report per-seed scores, mean and sample standard deviation (ddof=1). For paired comparisons, first average each user's paired difference across seeds, then bootstrap users with replacement 2,000 times, bootstrap seed 2026, percentile 95% intervals. This conditions on the split and fitted models; it is not a multi-dataset confidence interval. Descriptive rating-error/ranking scatter does not establish causality.

## Provenance and resource measurement

Every run records clean source commit, complete base and selected config, config hash, dataset fingerprint, split hashes, seed, UTC timestamp, environment versions, CPU identity, thread count, runtimes, process-lifetime peak RSS, loss curves and local checkpoint checksum. Reports export numeric evidence and anonymized aligned per-user metrics, not ratings or split rows. Reports' manifest maps file hashes to source runs. Evidence is separately committed after the source commit used for experiments. Exact floating-point reproducibility is only asserted within the tested CPU software environment, not arbitrary devices.

## References and licensing

- MovieLens: https://grouplens.org/datasets/movielens/ and https://files.grouplens.org/datasets/movielens/ml-1m-README.txt
- Harper and Konstan (2015), The MovieLens Datasets: History and Context, https://doi.org/10.1145/2827872
- Sedhain et al. (2015), AutoRec, https://doi.org/10.1145/2740908.2742726
- Rendle et al. (2009), BPR, https://proceedings.mlr.press/r7/rendle09a.html
- He et al. (2020), LightGCN, https://arxiv.org/abs/2002.02126

Code is independently implemented under MIT. MovieLens remains under GroupLens usage terms and is not included in this repository. No RecRanker code is copied, no affiliation is claimed, and LLM re-ranking is deferred.
