# Validation status

The local test suite covers chronological splitting, disjointness, deterministic ties/seeds, duplicate rejection, official-format parsing and checksums, full-catalog masking including low ratings, cold targets, tiny-case Recall/NDCG, empty populations, nonfinite scores, fit-only statistics, future-label-independent sampling, AutoRec observed-only gradients, LightGCN graph structure/zero-depth equivalence, all five checkpoint round trips and the validation-selection/final-refit/test boundary with an overwrite guard.

Run `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q` (disables unrelated host-installed pytest plugins). Exact local execution output is recorded in `test_output.txt`.

Five real MovieLens 100K one-epoch CPU smoke runs are executed separately; see `smoke_summary.csv` and the `smoke_*` run evidence. They validate execution, not convergence or headline performance.

GitHub Actions workflow: `.github/workflows/ci.yml`, with separate offline unit-test and official-download ML-100K jobs. **Hosted execution is not yet verified**; local validation must not be described as a passing remote CI run. Repository publishing status is reported in the handoff.

Additional completed checks: a new Python 3.12 virtual environment installed requirements-lock.txt successfully; a fresh local Git clone passed all 20 tests (4.39s); pip check found no broken requirements. All five real ML-100K smoke models completed. The real ML-1M partition audit verified disjoint, exhaustive partitions and timestamp boundaries for all 6,040 users. Main experiments and the sampling ablation completed all 18 planned final tests.
