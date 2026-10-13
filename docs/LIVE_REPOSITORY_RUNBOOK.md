# Live Repository Demo Runbook

## Purpose

The public demo localizes a natural-language issue in a public Python repository. It has no benchmark gold and therefore reports no accuracy, Recall, or MRR. It checks out source but never executes target code, installs target dependencies, or runs target setup scripts.

## Start

From the repository root with Python 3.11:

```powershell
.venv\Scripts\python.exe -m streamlit run scripts\public_demo.py
```

## Workflow

1. Choose the included example or enter a public GitHub repository URL.
2. Enter an issue title and description.
3. Optionally expand **Advanced settings** to pin a commit or select methods.
4. Select **Localize issue**.
5. Review the resolved commit, parsed structure, graph, retrieval rankings, and any enabled graph-guided stages.

The exact resolved commit is recorded with the result. Presentation-only widget changes reuse the saved session result and do not rerun inference.

## Method availability

- **BM25:** requires Git and an available source snapshot; it does not require a model.
- **Dense:** requires the configured sentence-transformer model to be locally available.
- **Hybrid:** requires both BM25 and Dense outputs.
- **Static Graph:** operates on available seed rankings and the parsed repository graph.
- **Adaptive:** requires the configured local Ollama model. The controller proposes bounded actions; deterministic code validates actions and reranks evidence.

If a dependency is unavailable, the interface reports it and does not present that stage as completed.

## Safety and scope

- Only public GitHub HTTPS repository URLs are accepted.
- Commits are resolved and pinned before parsing.
- Target repository code is treated as data and is not executed.
- The scope is Python file and callable localization, not patch generation.
- Live mode has no gold labels and makes no benchmark-accuracy claim.

## Troubleshooting

- Confirm Python is 3.11 with `scripts\verify_environment.py`.
- Confirm Git is available with `git --version`.
- Use BM25 if model-dependent methods are unavailable.
- A fresh repository needs network access for clone/fetch; a cached pinned snapshot can be replayed locally.
