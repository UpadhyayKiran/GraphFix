# GraphFix

GraphFix is a graph-guided software issue-localization system for Python repositories. Given an issue title and description, it ranks likely files and callables by combining lexical retrieval, dense retrieval, reciprocal-rank fusion, static repository-graph expansion, and bounded adaptive graph navigation.

The repository is a public, compact release of the completed Phase-1 implementation. It includes the implementation, tests, configuration, a live-repository demonstration, frozen cohort metadata, and concise saved evaluation evidence. Large historical repository caches, model caches, raw issue records, and per-instance bulk artifacts are intentionally excluded.

## Problem statement

Issue reports describe behavior, while repositories encode implementation details across files,
classes, methods, imports, calls, and inheritance relationships. GraphFix studies whether lexical
and semantic retrieval can be strengthened by navigating those structural relationships while
keeping ranking, action validation, and evaluation deterministic and auditable.

## Key engineering features

- Typed, modular Python package with separate parsing, graph, retrieval, navigation, pipeline,
  repository-management, and evaluation layers.
- Python AST extraction of modules, classes, functions, methods, signatures, and relationships.
- BM25, MiniLM dense retrieval, Hybrid reciprocal-rank fusion, and a static graph baseline.
- Prompted adaptive navigation through a bounded deterministic action environment.
- Persisted rankings and traces with post-hoc, gold-safe evaluation.
- Streamlit explorers for live repositories and saved benchmark evidence.

## Technology

Python 3.11, `ast`, Pydantic, GitPython, Hugging Face Datasets, BM25, Sentence Transformers,
typed repository-graph models, Ollama, Streamlit, pytest, Ruff, and mypy. Neo4j
export is optional and is not part of the core ranking pipeline.

## What GraphFix does

GraphFix supports two research workflows:

- **Live Repository localization** resolves and pins a public GitHub commit, checks out source without executing target code, parses Python structure, builds search documents and a repository graph, and presents ranked candidate files and callables. BM25 is available offline once the source snapshot exists; MiniLM and Ollama-backed methods require their documented external dependencies.
- **Benchmark evaluation** reconstructs benchmark-provided pre-fix snapshots and evaluates localization rankings only after inference is complete. Patch-derived gold labels are isolated from retrieval, graph traversal, and reranking.

The public Streamlit entry point exposes the gold-free live workflow. The complete academic benchmark explorer remains in `scripts/phase1_final_demo.py`, but its interactive benchmark mode requires the separately retained historical checkouts and bulk per-instance artifacts.

## Pipeline

```text
Issue title + description
        |
Pinned Python source snapshot
        |
AST parsing -> ParsedRepository
        |                 |
        |                 +-> SearchDocuments -> BM25 / Dense -> Hybrid RRF
        +-> RepositoryGraph                         |
                 |                                  +-> Adaptive navigation
                 +-> Static graph baseline          +-> Evidence reranking
                                      |
                              Ranked files and callables
                                      |
                         Post-hoc gold evaluation (benchmark only)
```

The prompted controller selects bounded graph actions; it does not directly emit the final ranking. A deterministic environment validates and executes actions, returns bounded observations, and an evidence-based reranker produces the adaptive ranking.

## Repository layout

```text
src/graphfix/       Core dataset, reconstruction, parsing, graph, retrieval,
                    localization, live-repository, and evaluation modules
scripts/            CLI campaigns, verification utilities, and Streamlit apps
tests/              Unit, integration, gold-safety, and demo coverage tests
pyproject.toml       Build metadata, dependencies, and tool configuration
data/               Frozen 560-issue cohort metadata and dataset instructions
results/            Compact aggregate metrics, adaptive outcomes, pilot
                    evidence, and ablation summaries
docs/               Implementation, methodology, reproducibility, and runbooks
```

## Installation

GraphFix requires Python `>=3.11,<3.12` and Git.

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -e ".[dev,explorer]"
```

If PowerShell script execution permits activation, you may instead run:

```powershell
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev,explorer]"
```

No permanent execution-policy change is required.

Add dense retrieval support when needed:

```powershell
.venv\Scripts\python.exe -m pip install -e ".[dense]"
```

Verify the environment:

```powershell
.venv\Scripts\python.exe scripts\verify_environment.py
.venv\Scripts\python.exe -m graphfix --help
```

## Run the public demonstration

```powershell
.venv\Scripts\python.exe -m streamlit run scripts\public_demo.py
```

The app accepts a public GitHub repository URL, issue title, and issue description. The example selector supplies an editable Budget-Aware AutoLLM issue. Localization never executes the target repository, installs its dependencies, or runs its setup scripts.

Live operation requires network access for a repository not already cached. Dense retrieval additionally requires a compatible local sentence-transformer model. Adaptive navigation requires the configured local Ollama model. Availability is reported in the interface; BM25 remains usable without either model.

For the ten-stage academic benchmark demonstration with the separately retained frozen artifact collection:

```powershell
.venv\Scripts\python.exe -m streamlit run scripts\phase1_final_demo.py
```

See [docs/LIVE_REPOSITORY_RUNBOOK.md](docs/LIVE_REPOSITORY_RUNBOOK.md) and [docs/PHASE1_REPRODUCIBILITY.md](docs/PHASE1_REPRODUCIBILITY.md).

## Command-line examples

```powershell
# Localize a benchmark instance with BM25
.venv\Scripts\python.exe -m graphfix localize vllm-project__vllm-9390

# Run the test and static-analysis suite
.venv\Scripts\python.exe -m pytest
.venv\Scripts\python.exe -m ruff check src tests scripts
.venv\Scripts\python.exe -m mypy src tests scripts
```

Campaign scripts and their options are documented in [docs/IMPLEMENTATION_GUIDE.md](docs/IMPLEMENTATION_GUIDE.md). Recomputing model-dependent experiments requires the dataset, reconstructed snapshots, model caches, and local services described in the reproducibility guide.

## Dataset and scientific boundaries

Phase-1 evaluation uses the 560-example `test` split of `czlll/Loc-Bench_V1`. The frozen instance IDs are included in `results/localization_benchmark/evaluation_cohort.json`; repository and base-commit provenance is retained in saved result records and can be rematerialized from the source dataset. Raw benchmark records are not redistributed in this public copy because the dataset card does not currently state a redistribution license; use `scripts/materialize_locbench.py` to retrieve them from the publisher.

Gold-bearing patch information is unavailable to localization methods. It is loaded only after rankings have been persisted, for post-hoc metrics. Tests exercise that boundary.

## Included evaluation evidence

The compact release includes:

- full-560 aggregate comparisons in `results/phase1_evaluation_summary/`;
- the full-560 Adaptive Fixed aggregate and per-instance outcome table in `results/adaptive_fixed_full560/`;
- pilot traces and efficiency summaries in `results/pilot_summary/`;
- bounded candidate-expansion ablation summaries in `results/ablations/`.

The verified full-cohort results are:

| Method | File Hit@1 | File MRR | Callable Hit@1 | Callable MRR | Instances |
|---|---:|---:|---:|---:|---:|
| BM25 | 0.3000 | 0.4152 | 0.2375 | 0.3132 | 560 |
| Dense | 0.1339 | 0.2182 | 0.2036 | 0.2840 | 560 |
| Hybrid RRF | 0.3321 | 0.4447 | 0.2821 | 0.3758 | 560 |
| Static Graph | 0.1643 | 0.3261 | 0.1339 | 0.2646 | 560 |
| Adaptive Fixed | 0.3946 | 0.5020 | 0.3429 | 0.4233 | 560 |

These are saved Phase-1 results, not a claim of universal superiority. The adaptive evaluation uses an ordered 560-instance cohort and a fixed candidate pool. Candidate expansion was retained as an ablation and did not replace the primary fixed-pool configuration.

See [results/README.md](results/README.md), [docs/PHASE1_FINAL_RESULTS.md](docs/PHASE1_FINAL_RESULTS.md), and [docs/PHASE1_ADAPTIVE_EVALUATION.md](docs/PHASE1_ADAPTIVE_EVALUATION.md).

## Reproducibility and provenance

- Exact benchmark instance IDs are versioned, and saved result records retain their base commits.
- Live runs resolve and record a full commit SHA.
- Saved artifacts retain configuration and provenance fields.
- Omitted historical checkouts can be reconstructed from their repository and commit mappings.
- The public copy contains no virtual environment, downloaded model, complete checkout cache, raw issue text, or Git object mirror.

Historical absolute paths inside frozen records are provenance, not active runtime dependencies. Details are in [docs/PHASE1_REPRODUCIBILITY.md](docs/PHASE1_REPRODUCIBILITY.md) and [docs/PHASE1_FROZEN_ARTIFACTS.md](docs/PHASE1_FROZEN_ARTIFACTS.md).

## Limitations

- Phase 1 targets Python source localization; it does not generate or apply patches.
- Dense and adaptive methods depend on external local models and can vary with model/runtime versions.
- Repository graphs are static approximations and do not capture all runtime behavior.
- Saved benchmark evidence supports audit and comparison, but full recomputation requires the omitted source and model caches.
- Adaptive navigation does not improve every issue; repeated actions and limited new evidence remain observed failure modes.

## Future work

Planned research directions include a smaller learned traversal controller, calibrated early
stopping, stronger evidence aggregation, and resource-efficiency evaluation. These are future
directions, not capabilities claimed by this Phase-1 release.

## References

- Loc-Bench V1 dataset: <https://huggingface.co/datasets/czlll/Loc-Bench_V1>
- LocAgent repository: <https://github.com/gersteinlab/LocAgent>
- Zhou et al., “LocAgent: Graph-Guided LLM Agents for Code Localization,” ACL 2025: <https://aclanthology.org/2025.acl-long.426/>

GraphFix acknowledges the Loc-Bench and LocAgent authors for their public research resources.

## License

GraphFix's original source code is licensed under the MIT License. See [LICENSE](LICENSE) for details. Third-party dependencies, datasets, and pretrained models remain subject to their respective licenses.

## Author

Kiran Upadhyay — M.Tech Data Science and Artificial Intelligence, PES University.
