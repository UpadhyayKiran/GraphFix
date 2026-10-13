# Loc-Bench V1 materialization

GraphFix evaluates on the 560-example `test` split of `czlll/Loc-Bench_V1`:

<https://huggingface.co/datasets/czlll/Loc-Bench_V1>

Raw records are not redistributed in this public repository because the published dataset card does not currently state a redistribution license. Materialize them locally with Python 3.11:

```powershell
.venv\Scripts\python.exe scripts\materialize_locbench.py
```

The script validates the expected schema and cohort size. Frozen instance identifiers are versioned in `results/localization_benchmark/evaluation_cohort.json`; saved result records retain repository and base-commit provenance so the evaluation remains auditable without bundling issue and patch content.

Patch-derived fields are gold data. Localization code must not read them before producing and persisting a ranking; they are reserved for post-hoc evaluation.
