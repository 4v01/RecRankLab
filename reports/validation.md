# Validation evidence

Local Python 3.12 CPU: **24 tests passed** (`test_output.txt`). This includes split integrity, leakage boundaries, tiny exact metrics, candidate masking, fit-only sampling, graph normalization/zero depth, EASE closed form/zero diagonal, item AutoRec denoising/checkpoints and adaptive search whose simulated test feedback contradicts validation. A completed signature is reused without repeating test evaluation.

All six real MovieLens 100K smoke checks completed (Popularity, BiasMF, AutoRec, BPR-MF, LightGCN, EASE). These are pipeline checks, not headline metrics. All 33 new ML-1M final runs completed from clean source `ab7798096d563205153f632d1029c59559cd10aa`. Historical v1 evidence is excluded.

`python scripts/audit_results.py` independently checks source history, config/data signatures, artifact hashes, validation-only choice, per-user metric means, seed means/SD, paired differences, shared populations and excluded data/checkpoints. Its recorded output is in `audit_output.txt`.

GitHub Actions runs the same unit/artifact audit plus six real-data CPU smokes. The complete v2 delivery commit `c16eea6b7fa7d52a605926383a3fe466132af221` passed both jobs: [run 42528199680](https://github.com/4v01/RecRankLab/actions/runs/42528199680). The later external benchmark adds its own metadata/hash verification step; final documentation commits also trigger CI.
