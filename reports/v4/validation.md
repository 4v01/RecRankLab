# V4 validation evidence

68 local tests passed (7.69 seconds), including independent native-SGD equations, sparse-vs-dense AutoRec training, external-data checkpoint reconstruction, stable top-K/candidate equivalence, stopping and split tests. The source preparation CI at a2b7dd8 passed: https://github.com/4v01/RecRankLab/actions/runs/42647372584 . Test output is preserved in test_output.txt.

Ten real ML-100K one-epoch configuration/engine smoke artifacts are retained in smoke_summary.json; these are correctness checks, not benchmark scores. Standard CI repeats tests, old-generation audits, official-data preparation, the V4 evidence audit when feasible work completes, and these smoke paths.

The final audit independently verifies forty actual final records, six explicitly blocked records, source/config/seed/data signatures, selected/refit epochs, blend dependencies, official SST/R², anonymous per-user arithmetic, aggregate means/SD, paired point differences and twenty context-stress cases. See audit_output.txt. The final delivery CI link is available from the repository Actions page and the delivered commit SHA.

Numerical experiment source commits:

- `4e531a0e016f563375794138aec34123244f8b4a`
- `774775c1970f2c455d47d4399b5cdf2c289a7f44`
- `a2b7dd8c05849a23afd34a0a90c71243cf06789f`
- `eac194cdce4fe411ec84354fe3d457a7a4c5d980`

Hosted training: https://github.com/4v01/RecRankLab/actions/runs/42648058543 . An earlier checkout-only attempt failed before training because a short SHA was interpreted as a branch: https://github.com/4v01/RecRankLab/actions/runs/42647433983 . No result or candidate was discarded for performance. Raw V2/V3 evidence remains unchanged.
