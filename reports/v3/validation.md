# V3 validation evidence

- Unit and mathematical/integration checks: 42 passed (6.57 seconds); see test_output.txt. This includes R² undefined/negative/perfect cases, SVD++ fit-only context and save/load checks, stopping/refit behavior and new ranking model checks.
- Ten declared experiment configurations each passed a one-epoch real official ML-100K smoke, seed17. See smoke_summary.csv for exact source commits and metrics. These smoke scores are not benchmark results.
- Thirty completed official ML-1M final runs: eight primary configurations × three seeds, one separately declared low-decay configuration × three, and one separately declared SVD++ configuration × three. The manifest binds clean source SHA, full configuration/selection, data fingerprint, split hashes, environment and seed to each immutable record.
- scripts/audit_v3.py verifies artifact hashes, source-snapshot configurations, candidate selection, checkpoint selection and final refit epochs, shared cohorts, convergence, per-user metric arithmetic, aggregate means/SD and paired differences. It recomputes R² SST from official MovieLens held-out ratings; prepare ML-1M first. See audit_output.txt for the actual final audit output.
- scripts/audit_results.py and reports/external_benchmark/verify.py continue to check the immutable v2 evidence and the separate external-source comparison.

GitHub Actions executes the 42 tests, both generation-specific evidence audits, external benchmark verification and real ML-100K smoke checks for the six original models plus all ten v3 configurations. The exact delivery commit and latest CI status are available at https://github.com/4v01/RecRankLab/actions . Local verification is not described as a remote CI pass before Actions finishes.

The experiments use source 964084262c25184b2ca363bc5ea5022887e16d79 (24 primary runs), 5b8bd6a86adcd7881c36282956436cf0b0362444 (3 low-decay runs) and 49cda6e052b1fbed5c45a2eba60f91ffb0f6a6aa (3 SVD++ runs). Reporting-source and experiment-source provenance remain separate.
