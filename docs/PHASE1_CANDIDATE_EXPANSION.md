# Phase-1 Candidate-Pool Expansion Experiment

## Question

The original Adaptive Navigation baseline collected graph evidence but reranked only the
original Hybrid Top-10. This experiment asks whether a bounded, gold-free mechanism for
admitting observed graph nodes improves localization on the fixed 20-instance pilot.

## Ranking rule

Candidate expansion is opt-in; `candidate_pool="fixed"` remains the default. In expanded
mode, GraphFix:

1. runs the same file and function navigation episodes using only `GraphFixInput`;
2. converts actually observed, type-compatible graph nodes into candidates;
3. removes IDs already present in Hybrid and deterministically deduplicates discoveries;
4. admits at most five lexicographically ordered discoveries per localization level;
5. orders evidence-supported original candidates first, preserving their Hybrid rank;
6. places evidence-supported discoveries next, with deterministic document-ID ties;
7. places unsupported original candidates last, preserving their Hybrid rank; and
8. truncates the result to the original pool size (Top-10 here).

Expanded candidates receive no learned or gold-derived score. Their stored score is zero
because ordering is determined by provenance, observed support, Hybrid prior, and stable
IDs—not by pretending graph evidence is a retrieval score. Each result records admitted
IDs in `expanded_candidate_ids`.

## Gold safety

Prompts, navigation, discovery, admission, reranking, and stopping do not receive patch,
labels, edit functions, added functions, or evaluation metrics. Results are persisted
before `GraphFixGold` is read. Gold is used only by the offline analyzer.

## Fixed cohort and configuration

The experiment used the first 20 IDs from the frozen 560-instance cohort, in unchanged
order. Configuration was `qwen2.5-coder:7b`, Ollama, temperature 0, seed 0, 120-second
timeout, ten accepted steps per file/function episode, two retries, eight observed nodes,
400-character snippets, and five maximum expanded candidates. All 20 instances completed
with zero failures.

## Paired results

The fixed-pool numbers below are a deterministic counterfactual computed from the exact
same newly saved navigation traces. Candidate expansion is the only difference.

| Level / metric | Hybrid | Adaptive Fixed Pool | Adaptive Expanded Pool |
|---|---:|---:|---:|
| File Hit@1 | 0.400000 | 0.400000 | 0.400000 |
| File Recall@5 | 0.330000 | 0.405000 | 0.405000 |
| File Recall@10 | 0.459167 | 0.459167 | 0.459167 |
| File MRR | 0.497917 | 0.530556 | 0.530556 |
| Function Hit@1 | 0.200000 | 0.200000 | 0.200000 |
| Function Recall@5 | 0.144583 | 0.134583 | 0.134583 |
| Function Recall@10 | 0.214583 | 0.214583 | 0.214583 |
| Function MRR | 0.282639 | 0.290972 | 0.278472 |

Expansion admitted five file candidates and 22 function/method candidates across nine
instances. Relative to the same-trace fixed pool, 38 of 40 file/function outcomes were
unchanged and two degraded; none improved. Relative to Hybrid, the expanded variant had
three improvements, 33 unchanged outcomes, and four degradations, but those improvements
came from evidence reranking already present in the fixed-pool method.

## vLLM-9390 verification

For `vllm-project__vllm-9390`, the generic mechanism admitted:

`function:benchmarks/benchmark_serving.py:benchmarks.benchmark_serving.main:572`

at expanded function rank 3. The node was not in the initial Hybrid Top-10. Its relevance
was established only afterward. The first relevant function rank remained 1 because a
different gold-relevant function was already first. File rank remained the adaptive
improvement from 6 to 1. The complete saved comparison and traces remain in the private
academic archive; the public n=20 and n=50 summaries are in `results/ablations/`.

## Representative cases

### Successful adaptive rescue: vLLM-9390

The issue concerns maximum concurrency in the benchmarking script. Hybrid placed the first
gold file at rank 6. Navigation inspected and expanded benchmark-serving entities; evidence
reranking moved the file to rank 1. Expansion additionally admitted `main:572` at function
rank 3. This demonstrates that the new mechanism removes the eligibility limitation, but
not an aggregate expansion gain because another relevant function was already rank 1.

### Unchanged case: Chainlit-1575

Hybrid file rank 1 and function rank 9 remained unchanged. The trace took seven accepted
steps and admitted no expanded candidate. This is a no-op case: graph navigation did not
produce evidence capable of changing the final locations.

### Corruption case: dgl-5240

The relevant file began at Hybrid rank 1 and ended at rank 2 after evidence reranking;
function gold remained outside the Top-10. Five observed function candidates were admitted,
but none rescued the missing function. The case illustrates structural-noise risk and the
cost of navigation when the lexical/semantic ranking is already correct.

### Expansion degradation: Meson-11366

The relevant function was rank 8 in both Hybrid and the same-trace fixed pool. Five observed
`NinjaBackend` methods entered ahead of unsupported originals and displaced the relevant
function beyond rank 10. This is the clearest evidence that observed graph proximity is not
equivalent to relevance.

### Repetitive exploration: transformers-22496

The two episodes consumed all 20 accepted steps, with extensive repeated action-target
pairs, while file rank 1 and function rank 3 did not change. This supports Phase-2 work on
evidence sufficiency and calibrated early stopping rather than simply increasing traversal.

## Resource efficiency

The expanded run used 309 accepted steps, 350 model calls, 41 retries, and 50 validation
failures. There were 90 summed unique-node observations, 214 repeated action-target steps
(69.26%), and only 63 steps with new evidence (20.39%). Thus 79.61% of accepted navigation
steps added no new evidence. Seven episodes stopped explicitly, 15 stopped after invalid
responses, and 18 reached the maximum step count.

Localization runtime totaled 1,185.79 seconds: mean 59.29 and median 54.34 seconds per
instance. The trace counts exactly match the historical fixed-pool pilot, confirming that
candidate-pool mode affects deterministic post-navigation reranking rather than controller
behavior.

## Decision

Bounded expansion solved the mechanical eligibility limitation and behaved correctly on
vLLM-9390, but it did not improve any paired pilot outcome and degraded two. It therefore
does **not** justify replacing the fixed-pool default. The expanded mode remains a
reproducible ablation and a foundation for future evidence-quality gating. The completed
n=50 ablation provides the larger follow-up evidence without changing this conclusion.

## n=50 candidate-expansion ablation

The n=50 candidate-expansion ablation completed with zero failures. Its 100 episodes used 846
accepted steps, 956 model calls, 110 retries, and 122 validation failures. Repetition was
70.33%; new-evidence rate was 21.28%; 78.72% of steps added no new evidence. Twenty-three
episodes stopped explicitly, 28 stopped after invalid responses, and 49 reached the step
cap. Localization runtime totaled 2,652.28 seconds (mean 53.05, median 50.17 per issue).

| Level / metric | Hybrid | Adaptive Fixed Pool | Adaptive Expanded Pool |
|---|---:|---:|---:|
| File Hit@1 | 0.360000 | 0.400000 | 0.400000 |
| File Recall@5 | 0.321905 | 0.371905 | 0.371905 |
| File Recall@10 | 0.454429 | 0.454429 | 0.450429 |
| File MRR | 0.473833 | 0.524913 | 0.521579 |
| Function Hit@1 | 0.280000 | 0.300000 | 0.300000 |
| Function Recall@5 | 0.157214 | 0.186548 | 0.182881 |
| Function Recall@10 | 0.249881 | 0.249881 | 0.261548 |
| Function MRR | 0.350056 | 0.377278 | 0.367635 |

Expansion admitted 73 candidates across 27 instances. Relative to fixed pool on identical
traces, 94 of 100 first-relevant-rank outcomes were unchanged and six degraded; none
improved. Function Recall@10 increased because expansion recovered additional relevant
functions below the first relevant item, but file Recall@10, function Recall@5, file MRR,
and function MRR decreased. The larger subset therefore reinforces the decision not to
replace fixed pool.

The subsequent full-560 Adaptive Fixed evaluation is the definitive Phase-1 Adaptive result.
The n=50 expanded-pool run remains a candidate-expansion ablation and does not replace the
fixed-pool primary configuration.
