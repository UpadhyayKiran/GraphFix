"""Tests for deterministic analysis of saved adaptive pilot artifacts."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest


def _module() -> ModuleType:
    path = Path(__file__).parents[1] / "scripts" / "analyze_adaptive_pilot.py"
    spec = importlib.util.spec_from_file_location("analyze_adaptive_pilot", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ANALYSIS = _module()


def _step(action: str, node_id: str, observed: list[dict[str, str]]) -> dict[str, object]:
    return {
        "action": action,
        "node_id": node_id,
        "observed_nodes": observed,
        "message": "observation",
        "validation_errors": [],
        "invalid_attempts": [],
        "model_calls": 1,
    }


def _observed(node_id: str, snippet: str = "source") -> dict[str, str]:
    return {
        "node_id": node_id,
        "node_type": "function",
        "file_path": "a.py",
        "name": "target",
        "relationship": "SELF",
        "direction": "self",
        "snippet": snippet,
    }


def test_pilot_cohort_resolution_is_ordered_and_deterministic(tmp_path: Path) -> None:
    cohort = tmp_path / "cohort.json"
    cohort.write_text(json.dumps({"included_ids": ["b", "a", "c"]}), encoding="utf-8")
    assert ANALYSIS.resolve_pilot_cohort(cohort, 2) == ("b", "a")
    assert ANALYSIS.resolve_pilot_cohort(cohort, 2) == ("b", "a")


def test_trace_aggregation_counts_novelty_repetition_and_stopping() -> None:
    navigation = {
        "actual_steps": 3,
        "model_calls": 4,
        "retry_count": 1,
        "validation_failures": 1,
        "stopping_reason": "explicit_stop",
        "stopped_by_controller": True,
        "trace": [
            _step("INSPECT_FUNCTION", "function:a.py:target:1", [_observed("n1")]),
            _step("INSPECT_FUNCTION", "function:a.py:target:1", [_observed("n1")]),
            _step("EXPAND_CALLEES", "function:a.py:target:1", [_observed("n2")]),
        ],
    }
    stats = ANALYSIS.navigation_stats(navigation)
    assert stats["new_evidence_yes"] == 2
    assert stats["new_evidence_no"] == 1
    assert stats["repeated_action_targets"] == 1
    assert stats["unique_nodes"] == 2
    assert stats["stopping_reason"] == "explicit_stop"


def test_rank_outcomes_and_interpretations_are_stable() -> None:
    assert ANALYSIS._rank_change(6, 2) == "IMPROVED"
    assert ANALYSIS._rank_change(1, 2) == "DEGRADED"
    assert ANALYSIS._rank_change(None, None) == "UNCHANGED"
    assert ANALYSIS._interpretation(2, "UNCHANGED") == "UNNECESSARY_AGENT"
    assert ANALYSIS._interpretation(None, "UNCHANGED") == "NO_EFFECT"


def test_candidate_pool_detection_finds_post_hoc_relevant_outside_node() -> None:
    node = _observed("function:a.py:pkg.target:1")
    payload = {
        "file_navigation": {"initial_ranking": [], "trace": []},
        "function_navigation": {
            "initial_ranking": [{"document_id": "function:b.py:other:1"}],
            "trace": [_step("INSPECT_FUNCTION", node["node_id"], [node])],
        },
    }
    count, relevant = ANALYSIS._outside_pool_diagnostics(
        payload, {"a.py"}, {"a.py:target"}
    )
    assert count == 1
    assert relevant == [
        {"node_id": "function:a.py:pkg.target:1", "gold_levels": ["file", "function"]}
    ]


def test_saved_localization_artifact_load_does_not_require_gold() -> None:
    root = Path(__file__).parents[1]
    path = (
        root
        / "results"
        / "adaptive_navigation"
        / "UXARRAY__uxarray-1117.adaptive.localization.json"
    )
    if not path.is_file():
        pytest.skip("bulk adaptive artifacts are not included in the compact release")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert "gold" not in payload
    assert payload["file_navigation"]["initial_ranking"]
    assert payload["function_navigation"]["trace"]


def test_frozen_pilot_metric_and_candidate_pool_aggregation() -> None:
    dataset = (
        Path(__file__).parents[1] / "data" / "raw" / "locbench_v1" / "dataset"
    )
    if not dataset.is_dir():
        pytest.skip("raw Loc-Bench materialization is not included in the compact release")
    rows, summary = ANALYSIS.analyze_pilot()
    assert len(rows) == 20
    assert summary["metrics"]["hybrid"]["file_hit_at_1"] == 0.4
    assert summary["metrics"]["adaptive"]["file_recall_at_5"] == 0.40499999999999997
    assert summary["navigation"]["combined"]["new_evidence_yes"] == 63
    assert summary["navigation"]["combined"]["repeated_action_targets"] == 214
    assert summary["navigation"]["combined"]["stopping_reasons"] == {
        "explicit_stop": 7,
        "invalid_response_stop": 15,
        "max_steps": 18,
    }
    assert summary["candidate_pool"]["gold_function_outside_hybrid_top10"] == 10
    assert summary["candidate_pool"]["gold_relevant_outside_pool_nodes"] == 1
