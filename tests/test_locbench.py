"""
Purpose:
    Verify locbench behavior without requiring network access.

Input:
    Small temporary repositories, records, models, or fixtures created by the tests.

Output:
    Passing assertions when the implementation follows the Review-1 contract.

Pipeline stage:
    Regression testing for the GraphFix implementation.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from copy import deepcopy
from typing import Any

import pytest

from graphfix.datasets import (
    LocBenchLoadError,
    LocBenchValidationError,
    load_locbench,
    normalize_locbench_record,
    validate_raw_schema,
)


def valid_record() -> dict[str, Any]:
    """Return a minimal row matching the observed Loc-Bench V1 schema."""
    return {
        "repo": "owner/repository",
        "instance_id": "owner__repository-1",
        "base_commit": "a" * 40,
        "patch": "full patch content",
        "test_patch": "test patch content",
        "problem_statement": "A reproducible issue report.",
        "hints_text": "",
        "created_at": 1_700_000_000,
        "labels": ["bug"],
        "category": "Bug Fix",
        "edit_functions": ["module.py:existing_function"],
        "added_functions": ["module.py:new_function"],
        "edit_functions_length": 1,
        "__index_level_0__": 0,
    }


class FakeDataset:
    """Network-free structural substitute for a Hugging Face Dataset."""

    def __init__(self, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> None:
        """Initialize the object and store the configuration needed by later operations."""
        self._rows = tuple(rows)
        self._columns = tuple(columns)

    @property
    def column_names(self) -> Sequence[str]:
        """Perform column names."""
        return self._columns

    def __iter__(self) -> Iterator[Mapping[str, Any]]:
        """Iterate over the stored records without changing them."""
        return iter(self._rows)


def test_valid_record_normalization_preserves_gold_values() -> None:
    """Verify valid record normalization preserves gold values."""
    raw = valid_record()
    instance, anomalies = normalize_locbench_record(raw)

    assert instance.input.instance_id == raw["instance_id"]
    assert instance.gold.patch == raw["patch"]
    assert instance.gold.edit_functions == raw["edit_functions"]
    assert instance.gold.edit_functions is not raw["edit_functions"]
    assert anomalies == ()


def test_missing_required_field_is_fatal() -> None:
    """Verify missing required field is fatal."""
    raw = valid_record()
    del raw["base_commit"]

    with pytest.raises(LocBenchValidationError, match="base_commit"):
        normalize_locbench_record(raw)


def test_empty_problem_statement_is_fatal() -> None:
    """Verify empty problem statement is fatal."""
    raw = valid_record()
    raw["problem_statement"] = "  "

    with pytest.raises(LocBenchValidationError, match="problem_statement"):
        normalize_locbench_record(raw)


def test_gold_function_fields_require_string_lists() -> None:
    """Verify gold function fields require string lists."""
    raw = valid_record()
    raw["edit_functions"] = "module.py:function"

    with pytest.raises(LocBenchValidationError, match="list of strings"):
        normalize_locbench_record(raw)


def test_null_labels_are_preserved_and_reported() -> None:
    """Verify null labels are preserved and reported."""
    raw = valid_record()
    raw["labels"] = None

    instance, anomalies = normalize_locbench_record(raw)

    assert instance.gold.labels is None
    assert [anomaly.code for anomaly in anomalies] == ["null_labels"]


def test_inference_iterator_excludes_all_gold_fields() -> None:
    """Verify inference iterator excludes all gold fields."""
    raw = valid_record()
    fake = FakeDataset([raw], tuple(raw))
    dataset = load_locbench(provider=lambda _name, _split: fake)

    inference_record = next(dataset.inference_inputs()).model_dump()
    forbidden = {"patch", "test_patch", "edit_functions", "added_functions", "labels", "gold"}
    assert forbidden.isdisjoint(inference_record)
    assert set(inference_record) == {
        "instance_id",
        "repo",
        "base_commit",
        "problem_statement",
        "category",
    }


def test_loader_limit_uses_injected_provider_without_network() -> None:
    """Verify loader limit uses injected provider without network."""
    rows = [deepcopy(valid_record()) for _ in range(3)]
    for index, row in enumerate(rows):
        row["instance_id"] = f"owner__repository-{index}"
    fake = FakeDataset(rows, tuple(rows[0]))

    dataset = load_locbench(limit=2, provider=lambda _name, _split: fake)

    assert len(dataset) == 2
    assert [item.instance_id for item in dataset.inference_inputs()] == [
        "owner__repository-0",
        "owner__repository-1",
    ]


def test_schema_validation_allows_extra_fields_but_rejects_missing_fields() -> None:
    """Verify schema validation allows extra fields but rejects missing fields."""
    raw = valid_record()
    observed = validate_raw_schema([*raw, "future_field"])
    assert "future_field" in observed

    with pytest.raises(LocBenchValidationError, match="labels"):
        validate_raw_schema([field for field in raw if field != "labels"])


def test_unusual_values_create_warnings_without_rewriting() -> None:
    """Verify unusual values create warnings without rewriting."""
    raw = valid_record()
    raw["repo"] = "not-a-pair"
    raw["base_commit"] = "not-a-normal-sha"
    raw["edit_functions_length"] = 9

    instance, anomalies = normalize_locbench_record(raw)

    assert instance.input.repo == "not-a-pair"
    assert instance.input.base_commit == "not-a-normal-sha"
    assert {anomaly.code for anomaly in anomalies} == {
        "repo_format",
        "sha_length",
        "sha_characters",
        "edit_function_count_mismatch",
    }


def test_loader_wraps_provider_and_validation_failures() -> None:
    """Verify loader wraps provider and validation failures."""
    def failing_provider(_name: str, _split: str) -> FakeDataset:
        """Perform failing provider."""
        raise OSError("offline")

    with pytest.raises(LocBenchLoadError, match="Unable to load"):
        load_locbench(provider=failing_provider)

    raw = valid_record()
    del raw["problem_statement"]
    fake = FakeDataset([raw], tuple(raw))
    with pytest.raises(LocBenchLoadError, match="Unable to normalize"):
        load_locbench(provider=lambda _name, _split: fake)
