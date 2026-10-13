# Phase-1 Frozen Artifacts

## Status

The Phase-1 evaluation artifacts are frozen scientific records. Public-repository preparation did not recompute localization, edit metrics, or rewrite stored rankings.

## Included in this compact release

- `results/phase1_evaluation_summary/`: full-560 deterministic and adaptive aggregate summaries.
- `results/adaptive_fixed_full560/`: Adaptive Fixed full-cohort summary, comparison, outcomes, and efficiency analysis.
- `results/pilot_summary/`: pilot summary, metrics, and efficiency figures.
- `results/ablations/`: n=20 and n=50 candidate-expansion summaries.
- `results/localization_benchmark/evaluation_cohort.json`: the frozen evaluation cohort.
- `data/raw/locbench_v1/manifest.json`: source materialization provenance and checksums.

## Retained outside the public repository

The private academic archive retains raw Loc-Bench records, all historical repository checkouts and Git mirrors, model/embedding caches, per-instance localization JSON for every method, complete adaptive traces, transient run logs, and submission-specific document packages.

These omissions reduce redistribution and repository-size risk. They do not change the reported scientific values. Full recomputation requires rematerializing or restoring these resources as described in `docs/PHASE1_REPRODUCIBILITY.md`.

## Integrity rule

Treat included result files as read-only. When publishing a release, record a Git commit or release checksum. Historical absolute paths embedded in frozen records are provenance only and should not be replaced with machine-specific public paths.
