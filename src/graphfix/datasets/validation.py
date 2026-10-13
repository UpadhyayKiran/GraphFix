"""
Purpose:
    Validates and normalizes raw Loc-Bench rows.

Input:
    Raw field names and one row mapping.

Output:
    A typed instance plus anomaly records.

Pipeline stage:
    Dataset ingestion.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict, ValidationError

from graphfix.datasets.models import (
    GraphFixGold,
    GraphFixInput,
    GraphFixInstance,
    GraphFixMetadata,
)

REQUIRED_RAW_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "repo",
        "instance_id",
        "base_commit",
        "patch",
        "test_patch",
        "problem_statement",
        "hints_text",
        "created_at",
        "labels",
        "category",
        "edit_functions",
        "added_functions",
        "edit_functions_length",
    }
)
_OWNER_REPOSITORY = re.compile(r"^[^/\s]+/[^/\s]+$")
_HEX_SHA = re.compile(r"^[0-9a-fA-F]+$")


class DatasetAnomaly(BaseModel):
    """A warning about source data that normalization deliberately preserves."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    instance_id: str
    code: str
    message: str
    field: str
    severity: Literal["warning"] = "warning"


class LocBenchValidationError(ValueError):
    """Raised when a row or schema cannot produce a safe normalized record."""


def validate_raw_schema(raw_fields: Sequence[str]) -> tuple[str, ...]:
    """Validate required columns and return all observed columns unchanged."""
    observed = tuple(raw_fields)
    missing = sorted(REQUIRED_RAW_FIELDS.difference(observed))
    if missing:
        joined = ", ".join(missing)
        raise LocBenchValidationError(f"Loc-Bench schema is missing required fields: {joined}")
    return observed


def _required_string(record: Mapping[str, Any], field: str) -> str:
    """
    Purpose:
        Read a required non-empty string field from a Loc-Bench row.

    Args:
        record: One Loc-Bench record.
        field: Name of the dataset field being processed.

    Returns:
        A `str` result produced by the operation.

    Raises:
        LocBenchValidationError: If the input or operation cannot be completed safely.

    Pipeline role:
        Dataset preparation, before any repository work.
    """
    value = record.get(field)
    if not isinstance(value, str):
        raise LocBenchValidationError(f"Field '{field}' must be a string")
    if field in {"instance_id", "repo", "base_commit", "problem_statement"} and not value.strip():
        raise LocBenchValidationError(f"Field '{field}' must be non-empty")
    return value


def _string_list(record: Mapping[str, Any], field: str) -> list[str]:
    """
    Purpose:
        Read a required list containing only strings.

    Args:
        record: One Loc-Bench record.
        field: Name of the dataset field being processed.

    Returns:
        A `list[str]` result produced by the operation.

    Raises:
        LocBenchValidationError: If the input or operation cannot be completed safely.

    Pipeline role:
        Dataset preparation, before any repository work.
    """
    value = record.get(field)
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise LocBenchValidationError(f"Field '{field}' must be a list of strings")
    return list(value)


def _optional_string_list(record: Mapping[str, Any], field: str) -> list[str] | None:
    """
    Purpose:
        Read an optional string-list field, treating null as an empty list.

    Args:
        record: One Loc-Bench record.
        field: Name of the dataset field being processed.

    Returns:
        A `list[str] | None` result produced by the operation.

    Pipeline role:
        Dataset preparation, before any repository work.
    """
    value = record.get(field)
    if value is None:
        return None
    return _string_list(record, field)


def _required_int(record: Mapping[str, Any], field: str) -> int:
    """
    Purpose:
        Read a required integer field while rejecting booleans.

    Args:
        record: One Loc-Bench record.
        field: Name of the dataset field being processed.

    Returns:
        A `int` result produced by the operation.

    Raises:
        LocBenchValidationError: If the input or operation cannot be completed safely.

    Pipeline role:
        Dataset preparation, before any repository work.
    """
    value = record.get(field)
    if not isinstance(value, int) or isinstance(value, bool):
        raise LocBenchValidationError(f"Field '{field}' must be an integer")
    return value


def _warning(instance_id: str, field: str, code: str, message: str) -> DatasetAnomaly:
    """
    Purpose:
        Create one structured, non-fatal dataset warning.

    Args:
        instance_id: The unique Loc-Bench instance ID.
        field: Name of the dataset field being processed.
        code: The code value used by this operation.
        message: The message value used by this operation.

    Returns:
        A `DatasetAnomaly` result produced by the operation.

    Pipeline role:
        Dataset preparation, before any repository work.
    """
    return DatasetAnomaly(instance_id=instance_id, field=field, code=code, message=message)


def normalize_locbench_record(
    record: Mapping[str, Any],
) -> tuple[GraphFixInstance, tuple[DatasetAnomaly, ...]]:
    """Normalize one raw row without rewriting benchmark values."""
    validate_raw_schema(tuple(record.keys()))
    instance_id = _required_string(record, "instance_id")
    repo = _required_string(record, "repo")
    base_commit = _required_string(record, "base_commit")
    problem_statement = _required_string(record, "problem_statement")
    category = _required_string(record, "category")
    edit_functions = _string_list(record, "edit_functions")
    added_functions = _string_list(record, "added_functions")
    labels = _optional_string_list(record, "labels")
    edit_functions_length = _required_int(record, "edit_functions_length")

    anomalies: list[DatasetAnomaly] = []
    if labels is None:
        anomalies.append(
            _warning(instance_id, "labels", "null_labels", "Gold labels are null")
        )
    if not _OWNER_REPOSITORY.fullmatch(repo):
        anomalies.append(
            _warning(instance_id, "repo", "repo_format", "Repository is not owner/repository-like")
        )
    if len(base_commit) != 40:
        anomalies.append(
            _warning(instance_id, "base_commit", "sha_length", "Commit SHA length is not 40")
        )
    if not _HEX_SHA.fullmatch(base_commit):
        anomalies.append(
            _warning(
                instance_id,
                "base_commit",
                "sha_characters",
                "Commit contains non-hex characters",
            )
        )
    if any(not function_name for function_name in edit_functions):
        anomalies.append(
            _warning(
                instance_id,
                "edit_functions",
                "empty_edit_function",
                "List contains an empty value",
            )
        )
    if any(not function_name for function_name in added_functions):
        anomalies.append(
            _warning(
                instance_id,
                "added_functions",
                "empty_added_function",
                "List contains an empty value",
            )
        )
    if edit_functions_length != len(edit_functions):
        anomalies.append(
            _warning(
                instance_id,
                "edit_functions_length",
                "edit_function_count_mismatch",
                "Recorded edit_functions_length does not match the list length",
            )
        )

    known_fields = REQUIRED_RAW_FIELDS | {"__index_level_0__"}
    extra_fields = {key: value for key, value in record.items() if key not in known_fields}
    try:
        instance = GraphFixInstance(
            input=GraphFixInput(
                instance_id=instance_id,
                repo=repo,
                base_commit=base_commit,
                problem_statement=problem_statement,
                category=category,
            ),
            gold=GraphFixGold(
                patch=_required_string(record, "patch"),
                test_patch=_required_string(record, "test_patch"),
                edit_functions=edit_functions,
                added_functions=added_functions,
                labels=labels,
            ),
            metadata=GraphFixMetadata(
                hints_text=_required_string(record, "hints_text"),
                created_at=_required_int(record, "created_at"),
                edit_functions_length=edit_functions_length,
                extra_fields=extra_fields,
            ),
        )
    except ValidationError as error:
        raise LocBenchValidationError(f"Invalid row '{instance_id}': {error}") from error
    return instance, tuple(anomalies)
