# Validation evidence

Local Python 3.12 CPU: **24 tests passed** (`test_output.txt`). This includes split integrity, leakage boundaries, tiny exact metrics, candidate masking, fit-only sampling, graph normalization/zero depth, EASE closed form/zero diagonal, item AutoRec denoising/checkpoints and adaptive search whose simulated test feedback contradicts validation. A completed signature is reused without repeating test evaluation.

All six real MovieLens 100K smoke checks completed (Popularity, BiasMF, AutoRec, BPR-MF, LightGCN, EASE). These are pipeline checks, not headline metrics. All 33 new ML-1M final runs completed from clean source `ab7798096d563205153f632d1029c59559cd10aa`. Historical v1 evidence is excluded.

`python scripts/audit_results.py` independently checks source history, config/data signatures, artifact hashes, validation-only choice, per-user metric means, seed means/SD, paired differences, shared populations and excluded data/checkpoints. Its recorded output is in `audit_output.txt`.

GitHub Actions runs the same unit/artifact audit plus six real-data CPU smokes. Remote status will be recorded after the first complete v2 push; no remote pass is claimed before execution.
