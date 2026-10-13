# Phase-1 Reproducibility

## Reproducibility levels

This compact public release supports three distinct levels of review:

1. **Implementation validation:** install the package and run tests, Ruff, and mypy.
2. **Saved-evidence audit:** inspect the frozen 560-instance cohort metadata, aggregate comparisons, adaptive per-instance outcomes, pilot evidence, and representative rankings included under `results/`.
3. **Full recomputation:** separately materialize Loc-Bench V1, reconstruct every pinned repository snapshot, provide model caches, and configure Ollama where required.

The public release does not imply that all 560 source checkouts are available offline.

## Environment

GraphFix requires Python `>=3.11,<3.12`.

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -e ".[dev,explorer]"
.venv\Scripts\python.exe scripts\verify_environment.py
```

For model-dependent workflows, install the optional dependencies declared in `pyproject.toml` and prepare the documented local models. Do not execute code from reconstructed target repositories.

## Dataset materialization

The evaluation uses `czlll/Loc-Bench_V1`, split `test`, with 560 frozen instance IDs. Raw records are intentionally not committed to this public repository.

```powershell
.venv\Scripts\python.exe scripts\materialize_locbench.py
```

The script downloads the dataset through Hugging Face, validates the expected schema, and
writes the local materialization. The authoritative instance-ID cohort is recorded in
`results/localization_benchmark/evaluation_cohort.json`. Saved result records retain their
repository and base-commit provenance; the full mapping can also be rematerialized from the
source dataset.

## Historical reconstruction

Each benchmark `base_commit` identifies the benchmark-provided pre-fix source snapshot. Reconstruction scripts clone/fetch the repository and create a detached snapshot at that commit. The mapping is provenance for reproducibility; it must not be described as the issue-creation-time commit unless independently established.

Large mirrors and checkout caches are omitted from the public release. Recreate them with the reconstruction utilities and configuration described in `docs/IMPLEMENTATION_GUIDE.md`.

## Validation

```powershell
.venv\Scripts\python.exe -m pytest
.venv\Scripts\python.exe -m ruff check src tests scripts
.venv\Scripts\python.exe -m mypy src tests scripts
```

`scripts/validate_phase1_demo_coverage.py` validates the complete academic artifact collection. It is expected to require the separately retained bulk per-instance results and representative historical checkout; it is not a compact-release completeness test.

## Gold-safety boundary

Localization receives issue text, reconstructed source, parsed structure, search documents, graph structure, and prior method rankings. Patch-derived gold data is loaded only after a localization result has been persisted. Gold is used for post-hoc evaluation and display below the explicit completion boundary. Gold-safety tests cover prohibited field access and evaluation sequencing.

## Saved evidence

The included result subset is documented in `results/README.md`. Full-560 aggregate rows and Adaptive Fixed per-instance outcomes permit count and metric consistency checks without rerunning inference. Bulk per-method localization JSON, complete traces, model caches, and all reconstructed repositories remain in the private academic archive.

Frozen JSON/CSV files must not be reformatted merely to update paths. Absolute paths inside them are historical provenance and are not active dependencies.
