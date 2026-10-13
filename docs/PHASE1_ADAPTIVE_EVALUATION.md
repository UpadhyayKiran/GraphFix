# Phase-1 Prompted Adaptive Navigation Evaluation

## Objective

This evaluation characterizes the benefits, failures, stopping behavior, and resource cost
of GraphFix's prompted graph-navigation baseline. It is a fixed 20-instance pilot, not a
claim of population-level superiority.

## Configuration and frozen cohort

The controller was local `qwen2.5-coder:7b` through Ollama at
`http://127.0.0.1:11434`, with temperature 0, seed 0, ten steps per localization level,
two bounded retries, and a 120-second controller timeout. All 20 cases completed with no
experiment-level failure or rerun.

The pilot uses, in frozen order, the first 20 IDs in
`results/localization_benchmark/evaluation_cohort.json`: UXARRAY-1117,
ultralytics-17810, Chainlit-1575, transformers-22496, scikit-learn-24145,
quiz-backend-84, openlibrary-7929, sisyphus-191, sympy-27223, modin-6836,
Open-MSS-1967, vllm-5473, tardis-2782, sopel-2285, meson-11366, dgl-5240,
pip-13085, simtools-1183, matplotlib-25887, and vllm-9390.

## Methodology and gold safety

The Adaptive pipeline received only `GraphFixInput`. Gold was unavailable to retrieval,
prompts, navigation, stopping, and reranking. Each gold-free result was persisted before
post-hoc evaluation loaded `GraphFixGold`. The reproducible analyzer reads the saved
artifacts and never invokes Ollama or localization.

`NEW_EVIDENCE=YES` means an accepted action added a previously unseen combination of graph
node, relationship, direction, and source observation. It does not mean gold relevance.
Repeated actions use the accepted `(action, target node)` pair.

## Hybrid versus Adaptive

| Level / metric | Hybrid | Adaptive |
|---|---:|---:|
| File Hit@1 | 0.4000 | 0.4000 |
| File Recall@5 | 0.3300 | 0.4050 |
| File Recall@10 | 0.4592 | 0.4592 |
| File MRR | 0.4979 | 0.5306 |
| Function Hit@1 | 0.2000 | 0.2000 |
| Function Recall@5 | 0.1446 | 0.1346 |
| Function Recall@10 | 0.2146 | 0.2146 |
| Function MRR | 0.2826 | 0.2910 |

Across 40 file/function outcomes, 3 improved, 35 were unchanged, and 2 degraded. The
diagnostic interpretation is 3 rescues, 2 corruptions, 18 unnecessary-agent outcomes, and
17 no-effect outcomes.

## Trace quality, stopping, and resources

The pilot used 309 accepted steps and 350 model calls, including 41 controller retries and
50 validation failures. Only 63 steps (20.39%) added new evidence; 246 (79.61%) did not.
There were 214 repeated action/target steps, a 69.26% repetition rate.

Across 40 navigation episodes, 7 (17.5%) ended with explicit `STOP`, 15 (37.5%) ended after
invalid responses, and 18 (45%) exhausted the step budget. Total runtime was 1,012.91
seconds: mean 50.65 and median 47.47 seconds per issue. The three improved localization
levels consumed 27 model calls, while most navigation left Hybrid rankings unchanged.

## Original candidate-pool limitation

Phase-1 Adaptive Navigation is an evidence-guided reranker over the original Hybrid Top
10. It cannot promote arbitrary graph-discovered nodes. Five cases had a gold file outside
Hybrid Top 10 and ten had a gold function outside it. Navigation observed 36 outside-pool
nodes in nine cases.

In `vllm-project__vllm-9390`, navigation reached the gold-relevant node
`function:benchmarks/benchmark_serving.py:benchmarks.benchmark_serving.main:572` outside
the pool. Post-hoc evaluation established its relevance, but the fixed-pool reranker could
not add it as a candidate. This was the fixed baseline evaluated for Review 2.

## Bounded expansion follow-up

Phase 1 now includes an opt-in bounded expanded-pool variant. On a new 20/20 run, the
generic mechanism admitted `main:572` at function rank 3. Across all 40 paired outcomes,
however, expansion improved none, left 38 unchanged, and degraded two relative to the
same-trace fixed pool. Function MRR fell from 0.290972 to 0.278472; file metrics were
unchanged. Fixed pool therefore remains the default. See
`docs/PHASE1_CANDIDATE_EXPANSION.md` for the rule, complete metrics, cases, and resource
analysis.

## Representative cases

- `vllm-project__vllm-9390`: file rank 6→1 and a gold-relevant outside-pool function;
  demonstrates both potential and the fixed-pool limitation.
- `scikit-learn__scikit-learn-24145`: file rank 6→2; a second positive reranking example.
- `dmlc__dgl-5240`: file rank 1→2; demonstrates corruption risk and structural noise.

## What worked and what failed

The controller interacted with real heterogeneous graph evidence, occasionally improved
ranking, and produced bounded, auditable traces. Its primary failures were repetitive
actions, low evidence novelty, weak explicit stopping, validation failures, and occasional
ranking corruption. Inability to promote newly discovered candidates is a limitation of the
primary fixed-pool configuration; the expanded-pool ablation removes that restriction but
introduces additional ranking noise.

## Conclusions and explicit non-claims

The historical n=20 pilot is descriptive and does not establish population-level
superiority. The definitive full-560 Adaptive Fixed evaluation improves Hit@1, Recall@5,
and MRR over Hybrid at file and callable levels while Recall@10 remains unchanged,
indicating improved prioritization within the fixed candidate pool. This cohort-specific
result does not establish global superiority. Expanded-pool Adaptive admits only bounded,
observed, type-compatible candidates and remains an ablation.

## Phase-1 closure status

Candidate-expansion ablations and failure/resource analyses are complete. Adaptive Fixed
remains the primary Phase-1 configuration, and the full-560 Adaptive Fixed evaluation is
the definitive Phase-1 Adaptive result. Phase-1 implementation and evaluation are complete.

The observed repetition, weak stopping, and unnecessary-agent cost motivate future
evidence-sufficiency-guided selective graph navigation and optional lightweight controller
adaptation. These are future research directions and are not part of the current evaluated
Phase-1 implementation.
