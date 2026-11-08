# RecRankLab v2: frozen protocol and improvement study

This study continues the supplied first version. The old test results were available before this study and informed the broad hypotheses. We do not claim a pristine, previously unseen benchmark. All v2 configurations and search rules are committed before running the new study; v2 test feedback is never used for candidate generation, selection or epoch changes.

## Data and shared protocol

MovieLens 1M is downloaded only from official GroupLens, 100K is a separate CPU smoke check. Raw ratings, archives, split indices, remapping arrays and checkpoints remain gitignored. Sort each user's rows by timestamp, item ID and source row; hold out roughly 80/10/10, using the identical original implementation. Hash source ZIP, rating member, split rows and processed files. The structural ID universe is known, but unsupported items are excluded from ranking. All fit statistics use only the current fit frame.

Rating observations and thresholded positives (rating >=4) are views of the same split. The candidate catalog is all items with at least one fit rating, excluding **all** of the user's prior fit ratings, including low ratings. Validation uses train history; test uses train+validation history. Candidates, eligible users and metrics are identical across models. Rank full catalog, stable ties by ascending item ID, macro Recall/NDCG at 10/20. Coverage is unique recommended items divided by supported catalog size. Excluded users and cold targets are counted. Rating RMSE/MAE cover all held-out observed ratings; cold predictions use fit mean and are not clipped.

Per-user chronology is not global forward chronology. Another user's fit events may be later in calendar time. Missing ratings are unobserved, not verified dislikes. The negative sampler takes only fit history and catalog. It cannot blacklist future positives without leaking future behavior; supported future positives may be sampled as unknown. The original brief's conflicting future-positive exclusion instruction is resolved in favor of fit-only information.

## Core baselines: unchanged budget

Popularity counts fit positives. BiasMF uses global mean, user/item biases and 32-dimensional factors, observed MSE plus sampled L2. User AutoRec uses 64 sigmoid hidden units and observed-only reconstruction MSE. BPR uses 32-dimensional embeddings, softplus pairwise loss and sampled initial-embedding L2. LightGCN averages layer 0/1/2 of a normalized positive-only bipartite graph and uses the same BPR objective/sampler. Each core trainable model gets two learning rates and 20 epochs; exact settings are in configs/base.yaml and model YAMLs. The positives-only BPR sampler ablation has the same budget. This isolates a small-budget comparison, not convergence.

## Improvement study: separately disclosed larger budgets

The machine-readable frozen plan is configs/study.yaml; seeds are 17,29,43. Optimized BPR has 64 factors, 80 epochs, smaller batches and four learning-rate/regularization combinations. Optimized LightGCN has 64 factors, 60 epochs, larger batches to bound graph computation, and four learning-rate/depth combinations. Item AutoRec has 128 hidden units, 60 epochs, two learning rates crossed with input dropout 0/.2. Corruption is input-only; the loss always supervises the original observed ratings. These are combined implementation/budget changes, not causal one-factor ablations. Report actual fits and runtime, and never describe their larger search budget as equal to core baselines.

EASE (Steck, WWW 2019, https://doi.org/10.1145/3308558.3313710) is independently implemented on binary fit positives: G=X'X+lambda I, P=G^-1, B_ij=-P_ij/P_jj, diag(B)=0, scores=XB. Use float64 Cholesky solve, store coefficients in float32. The four fixed ridge candidates are 30,100,300,1000. The model has no gradient-trained parameters, but stores a full item-item coefficient matrix; report that memory limitation and score time.

## Bounded self-optimization

The equal-four-fit EASE search ablation uses anchors 100/1000, then divides or multiplies the validation elite's ridge by 3, avoiding duplicates and bounding ridge to [1,10000]. Both successes and failures remain in a per-run validation memory with hypothesis, parent, config and fit time. The finite fallback list is committed in search.py. Grid and memory search use the same model and four validation fits. All seeds use the same policy; EASE is deterministic, so repeated seeds are a reproducibility check, not three independent randomized fits. The module accepts only config/validation records; no test frame or test evaluator is accessible inside it. It does not generate or execute arbitrary code and does not call an LLM.

This limited mechanism is inspired by the hypothesis/validation/memory loop described on the supplied GIM research pages, not a reproduction of CogAlpha or XALPHA. Public pages were read on 2026-11-09 (Asia/Shanghai):
- https://graceim.ai/zh/research/cognitive-alpha-mining
- https://graceim.ai/zh/research/xalpha-memory-driven-ai-quant-researcher

No financial data, performance numbers, code or affiliations are imported. No claim about GIM's published performance is independently verified here. LLM reranking remains deferred.

## Selection, refit, inference and uncertainty

Reset random states before every trial. Select maximum validation NDCG@10, first candidate wins ties. Persist selection, reinitialize and fit train+validation for the selected fixed epochs, then evaluate test exactly once per seed. No test-guided early stopping. Resume only a matching completed signature; reject incompatible overwrites. Never pool historical v1 and new v2 results.

Record clean Git commit, config hashes, complete parameters, dataset/split fingerprints, seeds, package versions, hardware, UTC time, train/evaluation/total seconds and process-lifetime RSS. The latter includes previous trials/seeds and is not per-model allocated memory. Final-fit seconds exclude tuning and serialization. Export anonymous aligned per-user metrics and checksums; checkpoints remain local. Report-generation code has its own commit.

Mean and sample SD are across seeds. Paired intervals average each user's seedwise difference, then bootstrap users 2,000 times (seed 2026, percentile 95%). Intervals condition on models and split, and omit search uncertainty/multiple comparisons. They do not correct for prior exposure to this benchmark.
