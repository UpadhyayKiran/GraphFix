# GraphFix Phase-1 Final Results

Definitive evidence: frozen 560-instance Loc-Bench V1 cohort. Adaptive uses the fixed Hybrid
Top-10 candidate pool. Historical n=20/n=50/n=100 evidence is retained but is not primary.

## Full-560 comparison

| Method | File H@1 | File R@5 | File R@10 | File MRR | Function H@1 | Function R@5 | Function R@10 | Function MRR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BM25 | 0.300000 | 0.485566 | 0.576424 | 0.415227 | 0.237500 | 0.302017 | 0.377924 | 0.313186 |
| Dense | 0.133929 | 0.257790 | 0.376313 | 0.218209 | 0.203571 | 0.261045 | 0.314697 | 0.284007 |
| Hybrid | 0.332143 | 0.489914 | 0.603865 | 0.444745 | 0.282143 | 0.356593 | 0.449396 | 0.375820 |
| Static Graph | 0.164286 | 0.458669 | 0.588433 | 0.326109 | 0.133929 | 0.317008 | 0.415331 | 0.264573 |
| Adaptive Fixed | **0.394643** | **0.519973** | **0.603865** | **0.501988** | **0.342857** | **0.385049** | **0.449396** | **0.423256** |

Hybrid is the strongest deterministic retrieval configuration. Adaptive Fixed achieves the
strongest ranking performance among evaluated methods on this cohort. Relative to Hybrid,
file deltas are +0.062500 H@1, +0.030060 R@5, 0 R@10, and +0.057243 MRR; function deltas are
+0.060714, +0.028456, 0, and +0.047436. Unchanged R@10 shows reranking within, not coverage
beyond, the fixed pool.

## Outcomes and efficiency

- File: 73 improved, 469 unchanged, 18 degraded.
- Function: 65 improved, 466 unchanged, 29 degraded.
- Combined: 138 improved, 935 unchanged, 47 degraded.
- 560 issues, 1,120 episodes, 8,677 accepted steps, 10,112 controller calls.
- 1,435 retries and 1,573 validation failures.
- 6,105 repeated action-targets (70.36%); 1,665 new-evidence steps (19.19%).
- 233 explicit stops (20.80%), 420 invalid-response stops (37.50%), and 467 maximum-step
  terminations (41.70%).
- Persisted runtime 35,971.87 seconds; mean 64.24 and median 48.95 seconds per issue.

## Historical expansion ablations

- Expanded n=20 versus same-trace fixed: 0 improved, 38 unchanged, 2 degraded.
- n=50 candidate-expansion ablation: 0 improved, 94 unchanged, 6 degraded.

Expansion enables outside-pool candidates but did not improve first-relevant rank and caused
degradations. Fixed pool remains primary.

## Conclusion

Phase 1 demonstrates cohort-specific ranking gains from adaptive graph navigation, together
with substantial cost, repeated actions, low evidence novelty, and weak explicit stopping.
This motivates Phase-2 evidence-sufficiency-guided selective navigation. Phase 2 and QLoRA
are not implemented in these results.
