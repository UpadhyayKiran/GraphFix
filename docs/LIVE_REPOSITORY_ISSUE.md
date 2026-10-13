# Live Repository Demonstration Issue

## Repository state

- Repository: `https://github.com/UpadhyayKiran/Budget-Aware-AutoLLM`
- Default branch observed on 9 October 2026: `main`
- Pinned commit: `77aaf1b037bdf48fdec26385220279418e737809`
- Status: local demonstration request only; no GitHub issue was published and the target repository was not modified.

## Title

Support scientific notation in numeric answer extraction

## Description

Numeric answer evaluation currently performs partial extraction when a marked answer uses scientific notation. A response containing `Final answer: 1e3` is extracted and normalized as `1`, rather than `1000`; `Final answer: 2.5e-2` is extracted and normalized as `2.5`, rather than `0.025`.

Please extend the supported numeric formats to include positive and negative decimal exponents while preserving the existing decimal, fraction, percentage, and marked-answer precedence behavior.

## Minimal reproduction

At the pinned commit, the current `NUMBER_TOKEN` and `extract_numeric_answer` definitions were exercised directly in an isolated namespace without installing or executing the repository application:

| Input | Observed raw extraction | Observed normalized value | Expected value after the requested extension |
|---|---:|---:|---:|
| `Final answer: 1e3` | `1` | `1` | `1000` |
| `Final answer: 2.5e-2` | `2.5` | `2.5` | `0.025` |

This is therefore described as a requested extension to the extractor's supported formats. The present behavior is partial extraction: the existing numeric token ends before the exponent instead of treating the whole scientific-notation token as invalid or supported.

## Independent verification boundary

The previously suspected location, `src/budget_autollm/evaluation/numeric.py` and callable `extract_numeric_answer`, is recorded here only for post-run verification. It is not included in the issue text submitted to GraphFix and must not be passed as gold, prompt context, ranking input, or candidate-selection guidance.
