# GraphFix Implementation Walkthrough

## Pipeline map

`Loc-Bench → GraphFixInput → RepositoryManager → PythonRepositoryParser → ParsedRepository → RepositoryGraph/SearchDocument → localization ranking → gold extraction → metrics`

`GraphFixInput` and `GraphFixGold` are separate strict models. Reconstruction, parsing,
graph construction, document generation, and retrieval accept no gold data.

Each stage consumes the output immediately to its left: the dataset layer yields a
`GraphFixInput`; reconstruction yields a verified checkout; parsing yields a
`ParsedRepository`; graph/document builders yield `RepositoryGraph` and
`SearchDocument` collections; retrieval and navigation yield gold-free rankings; and the
evaluation layer combines those saved rankings with `GraphFixGold` only afterward.

## Dataset and reconstruction

- `datasets/locbench.py` loads and normalizes Loc-Bench.
- `datasets/models.py` defines `GraphFixInput`, `GraphFixGold`, metadata, and instances.
- `datasets/validation.py` validates the raw schema and records non-fatal anomalies.
- `repositories/manager.py::RepositoryManager.reconstruct` uses shared bare objects and
  isolated detached worktrees, then verifies `HEAD == base_commit`.
- `repositories/git.py::run_git` uses bounded argument-list Git execution with no shell.

## Parsing and graph construction

- `parsing/parser.py::PythonRepositoryParser` parses safe regular Python files without
  importing repository code. Per-file syntax, Unicode, recursion, and I/O failures are
  retained instead of aborting the repository.
- `parsing/models.py::ParsedRepository` stores files, symbols, failures, and statistics.
- `graph/builder.py::build_repository_graph` builds repository, directory, file, class,
  function, and method nodes.
- Edge types are `CONTAINS`, `DEFINED_IN`, `IMPORTS`, `CALLS`, and `INHERITS`.

## Search documents and retrieval

- `retrieval/documents.py` creates separate file and callable `SearchDocument` objects.
- `retrieval/bm25.py` provides code-aware tokenization and explicit Okapi BM25 with
  `k1=1.5`, `b=0.75`, descending score, and document-ID tie-breaking.
- `retrieval/dense.py` uses `sentence-transformers/all-MiniLM-L6-v2` and explicit cosine
  similarity. Its embedding cache is keyed by repository, exact commit, model, document
  level, count, and ordered document-content fingerprint.
- `retrieval/hybrid.py` implements RRF as `sum(1 / (k + rank))`, with `k=60`.
- `pipeline/advanced.py` ranks every available file/function document with BM25 and Dense,
  fuses the complete rankings, and only then returns Top-10 Hybrid results.

## Static Graph baseline

`retrieval/graph_baseline.py` takes the first 10 Hybrid candidates that resolve to graph
nodes. It traverses `CALLS`, `IMPORTS`, and `INHERITS` in both directions to depth 1.
Evidence is ordered by minimum hop distance, seed rank, then document ID. Reached callable
nodes remain callable candidates; other nodes with file paths may project to file documents.
The retrieval and graph rankings are fused with RRF (`k=60`) and truncated to Top-10.

The RepositoryGraph is built and inspectable. BM25 does not use graph edges for ranking.

## Pipelines and evaluation

- `pipeline/localizer.py::BM25LocalizationPipeline.localize` produces a gold-free BM25
  `LocalizationResult`. `Review1Pipeline` remains temporarily as a compatibility alias.
- `pipeline/advanced.py::AdvancedLocalizationPipeline.localize` produces gold-free Dense,
  Hybrid, or Static Graph results.
- `evaluation/matching.py` normalizes patch/function targets after localization.
- `evaluation/metrics.py` calculates file/function Hit@1, Recall@5, Recall@10, and MRR.
- `pipeline/campaign.py` supports resumable BM25 readiness, cohort, and evaluation runs.
- `scripts/run_retrieval_comparison.py` checkpoints each method independently and saves a
  localization result before accessing gold.

## Adaptive Navigation

`pipeline/adaptive.py::AdaptiveNavigationPipeline` obtains corrected Hybrid Top-10 initial
file/function rankings, reconstructs the verified graph context, and runs separate file and
function episodes. It accepts only `GraphFixInput`.

The exact action space is `INSPECT_FILE`, `INSPECT_FUNCTION`, `EXPAND_CALLERS`,
`EXPAND_CALLEES`, `EXPAND_IMPORTS`, `EXPAND_INHERITANCE`, and `STOP`. Calls and callers use
outgoing and incoming `CALLS`; imports use outgoing `IMPORTS`; inheritance uses outgoing
`INHERITS`.

`PromptedController` receives a deterministically structured prompt containing the issue,
localization level, initial candidate IDs/ranks/metadata/snippets, accumulated evidence,
previous action/observation history, remaining steps, allowed actions, and required JSON
schema. Snippets default to 400 characters and observations to eight reached nodes.

`agent/models.py::OllamaTextGenerator` provides a local provider-neutral callable adapter.
The model name is required externally; endpoint, timeout, temperature, and seed are recorded.
GraphFix never starts or downloads a model. Two invalid-response retries are allowed by
default; exhaustion safely records STOP. Episodes explicitly stop or terminate at 10 steps.

Evidence reranking is conservative: an original candidate is supported when its document
node or containing file node was observed. Supported candidates precede unsupported ones;
original Hybrid rank breaks order within each group. No learned score or gold data is used.

`AdaptiveNavigationConfig.candidate_pool` selects `fixed` (default) or `expanded`. Expanded
mode deterministically converts actually observed type-compatible graph nodes into at most
`max_expanded_candidates` new candidates per level, deduplicates them, places them after
evidence-supported original candidates, and truncates to the original pool size. Admitted
IDs are persisted in `AdaptiveLevelResult.expanded_candidate_ids`. The fixed behavior is
unchanged and remains the default because the n=20 expansion experiment did not improve a
paired outcome.

`AdaptiveLocalizationResult` stores both initial/final rankings, full bounded traces,
validation errors, model calls, retries, observed nodes, stopping reasons, runtime, graph and
parser statistics, provider/model identity, and controller settings. The pilot saves this
gold-free result before evaluation accesses `GraphFixGold`.

QLoRA, learned traversal, calibrated evidence-sufficiency stopping, and full
resource-efficiency analysis are not implemented.

## Operational scripts

- `materialize_locbench.py` — validate and materialize the local dataset snapshot.
- `inspect_locbench.py` — inspect schema, counts, anomalies, and safe input.
- `reconstruct_locbench_repos.py` — reconstruct a bounded repository set.
- `run_bm25_campaign.py` — BM25 readiness, cohort, and full evaluation.
- `verify_bm25_results.py` — independently verify 560 migrated BM25 artifacts.
- `run_retrieval_pilot.py` — bounded retrieval comparison.
- `run_retrieval_comparison.py` — resumable full retrieval comparison.
- `verify_retrieval_comparison.py` — read-only verification of the 560×4 comparison.
- `review1_explorer.py` — retained academic demo filename with project-semantic UI.
- `phase1_pipeline_explorer.py` — offline pipeline explorer used by the final demonstration.
- `analyze_fixed_adaptive.py` — post-hoc metrics, rank outcomes, and resource aggregation
  for a saved fixed-pool Adaptive cohort.
- `analyze_adaptive_pilot.py` — rebuild original fixed-pool pilot summaries.
- `analyze_candidate_expansion.py` — compare Hybrid, same-trace fixed pool, and expanded
  pool from saved results without Ollama.
- `verify_environment.py` — check the supported Python version, package import, and Git.

## Public result layout

- `results/localization_benchmark/evaluation_cohort.json` — frozen 560-instance cohort.
- `results/phase1_evaluation_summary/` — full-cohort aggregate comparison.
- `results/adaptive_fixed_full560/` — Adaptive Fixed aggregate and per-instance outcomes.
- `results/examples/vllm-project__vllm-9390/` — representative method results.
- `results/pilot_summary/` — pilot metrics and navigation-efficiency figures.
- `results/ablations/` — bounded candidate-expansion summaries.

The private academic archive retains the bulk per-instance result trees, complete traces,
run logs, and reconstructed repositories. Their omission from the public repository is
documented in `results/README.md` and `docs/PHASE1_REPRODUCIBILITY.md`.
