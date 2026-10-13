"""
Purpose:
    Verify materialize locbench behavior without requiring network access.

Input:
    Small temporary repositories, records, models, or fixtures created by the tests.

Output:
    Passing assertions when the implementation follows the Review-1 contract.

Pipeline stage:
    Regression testing for the GraphFix implementation.
"""

from __future__ import annotations

import csv
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pytest
from datasets import Dataset, load_from_disk  # type: ignore[import-untyped]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from graphfix.datasets import GraphFixGold, GraphFixInput  # noqa: E402
from materialize_locbench import (  # noqa: E402
    BROWSE_EXPORT_NAME,
    FULL_EXPORT_NAME,
    MaterializableDataset,
    materialize_dataset,
    resolve_output_dir,
    validate_dataset,
)

PATCH_SENTINEL = "RAW_PATCH_SENTINEL_DO_NOT_TRACK"
TEST_PATCH_SENTINEL = "RAW_TEST_PATCH_SENTINEL_DO_NOT_TRACK"


def tiny_dataset() -> MaterializableDataset:
    """Build a two-row Hugging Face dataset without network access."""
    dataset = Dataset.from_dict(
        {
            "repo": ["owner/repo", "owner/repo"],
            "instance_id": ["example-1", "example-2"],
            "base_commit": ["a" * 40, "b" * 40],
            "patch": [PATCH_SENTINEL, "second patch"],
            "test_patch": [TEST_PATCH_SENTINEL, "second test patch"],
            "problem_statement": ["First issue", "Second issue"],
            "hints_text": ["", "hint"],
            "created_at": [1, 2],
            "labels": [["src/example.py"], None],
            "category": ["bug", "bug"],
            "edit_functions": [["example.fix"], ["example.other"]],
            "added_functions": [[], ["example.new"]],
            "edit_functions_length": [1, 1],
            "__index_level_0__": [0, 1],
        }
    )
    return cast(MaterializableDataset, dataset)


def test_materialization_round_trip_and_exports_preserve_required_fields(tmp_path: Path) -> None:
    """Verify materialization round trip and exports preserve required fields."""
    raw_root = tmp_path / "data" / "raw"
    output_dir = raw_root / "locbench_v1"
    result = materialize_dataset(
        tiny_dataset(),
        output_dir=output_dir,
        cache_dir=tmp_path / "hf-cache",
        expected_rows=2,
        materialized_at=datetime(2026, 8, 22, tzinfo=UTC),
        raw_data_root=raw_root,
    )

    reloaded = load_from_disk(str(result.snapshot_path))
    assert len(reloaded) == 2
    assert tuple(reloaded.column_names) == result.column_names
    assert reloaded[0]["patch"] == PATCH_SENTINEL

    full_rows = [
        json.loads(line)
        for line in result.full_jsonl_path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(full_rows) == 2
    assert tuple(full_rows[0]) == result.column_names
    assert full_rows[0]["patch"] == PATCH_SENTINEL
    assert full_rows[0]["test_patch"] == TEST_PATCH_SENTINEL

    with result.browse_csv_path.open(encoding="utf-8-sig", newline="") as stream:
        browse_rows = list(csv.DictReader(stream))
    assert len(browse_rows) == 2
    assert "patch" not in browse_rows[0]
    assert "test_patch" not in browse_rows[0]
    assert browse_rows[0]["edit_functions"] == '["example.fix"]'
    assert browse_rows[0]["labels"] == '["src/example.py"]'

    manifest_text = result.manifest_path.read_text(encoding="utf-8")
    assert PATCH_SENTINEL not in manifest_text
    assert TEST_PATCH_SENTINEL not in manifest_text
    manifest = cast(dict[str, Any], json.loads(manifest_text))
    assert manifest["row_count"] == 2
    assert manifest["exports"]["full_jsonl"]["filename"] == FULL_EXPORT_NAME
    assert manifest["exports"]["browse_csv"]["filename"] == BROWSE_EXPORT_NAME


def test_materialization_requires_explicit_overwrite(tmp_path: Path) -> None:
    """Verify materialization requires explicit overwrite."""
    raw_root = tmp_path / "data" / "raw"
    output_dir = raw_root / "locbench_v1"
    materialize_dataset(
        tiny_dataset(),
        output_dir=output_dir,
        cache_dir=tmp_path / "hf-cache",
        expected_rows=2,
        raw_data_root=raw_root,
    )

    with pytest.raises(FileExistsError, match="--overwrite"):
        materialize_dataset(
            tiny_dataset(),
            output_dir=output_dir,
            cache_dir=tmp_path / "hf-cache",
            expected_rows=2,
            raw_data_root=raw_root,
        )


def test_row_count_and_output_boundary_validation(tmp_path: Path) -> None:
    """Verify row count and output boundary validation."""
    with pytest.raises(ValueError, match="Expected 3 rows, found 2"):
        validate_dataset(tiny_dataset(), expected_rows=3)

    raw_root = tmp_path / "data" / "raw"
    assert resolve_output_dir(raw_root / "locbench_v1", raw_data_root=raw_root).is_relative_to(
        raw_root
    )
    with pytest.raises(ValueError, match="must be a child"):
        resolve_output_dir(tmp_path / "outside", raw_data_root=raw_root)


def test_gold_fields_remain_outside_inference_model() -> None:
    """Verify gold fields remain outside inference model."""
    inference_fields = set(GraphFixInput.model_fields)
    gold_fields = set(GraphFixGold.model_fields)

    assert {"patch", "test_patch", "edit_functions", "added_functions", "labels"} <= gold_fields
    assert inference_fields.isdisjoint(gold_fields)
