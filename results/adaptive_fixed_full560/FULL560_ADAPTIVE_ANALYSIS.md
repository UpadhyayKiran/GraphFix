# Full-560 Adaptive Fixed-Pool Analysis

Generated: 2026-10-09T08:55:08.984407Z

All 560 artifacts exactly match the frozen cohort. The candidate-pool field is `adaptive_config.candidate_pool`; every artifact is `fixed`. Gold was loaded only for post-hoc evaluation.

## Metrics

| Method | File H@1 | File R@5 | File R@10 | File MRR | Function H@1 | Function R@5 | Function R@10 | Function MRR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bm25 | 0.300000 | 0.485566 | 0.576424 | 0.415227 | 0.237500 | 0.302017 | 0.377924 | 0.313186 |
| dense | 0.133929 | 0.257790 | 0.376313 | 0.218209 | 0.203571 | 0.261045 | 0.314697 | 0.284007 |
| hybrid | 0.332143 | 0.489914 | 0.603865 | 0.444745 | 0.282143 | 0.356593 | 0.449396 | 0.375820 |
| static_graph | 0.164286 | 0.458669 | 0.588433 | 0.326109 | 0.133929 | 0.317008 | 0.415331 | 0.264573 |
| adaptive_fixed | 0.394643 | 0.519973 | 0.603865 | 0.501988 | 0.342857 | 0.385049 | 0.449396 | 0.423256 |

## Hybrid to Adaptive Fixed outcomes

Lower first relevant rank is improved; higher is degraded; equal (including both absent from Top-10) is unchanged. Combined sums the file and function classifications.

- File: {'improved': 73, 'unchanged': 469, 'degraded': 18}
- Function: {'improved': 65, 'unchanged': 466, 'degraded': 29}
- Combined: {'improved': 138, 'unchanged': 935, 'degraded': 47}

## Efficiency

- Episodes: 1120; accepted steps: 8677; controller calls: 10112
- Repeated action-target rate: 0.703584; new-evidence rate: 0.191887
- Persisted runtime: 35971.871 seconds; mean: 64.235; median: 48.949

## Interpretation

Adaptive Fixed is a prompted reranker of the frozen Hybrid Top-10 candidate pool. The full-cohort evidence should be described metric by metric: it changes first-rank quality and MRR while leaving the candidate-pool ceiling unchanged. Its controller calls, repeated actions, validation failures, and runtime are additional resource costs. These results support conclusions for this frozen Loc-Bench cohort and configuration only, not global superiority.

Adaptive metric record: `{"file_hit_at_1": 0.39464285714285713, "file_mrr": 0.501988378684807, "file_recall_at_10": 0.6038648603380746, "file_recall_at_5": 0.5199730725623583, "function_hit_at_1": 0.34285714285714286, "function_mrr": 0.42325609410430826, "function_recall_at_10": 0.4493964437718434, "function_recall_at_5": 0.3850493905566179, "instances": 560}`
