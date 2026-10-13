"""
Purpose:
    Defines strict models that enforce the inference/gold leakage boundary.

Input:
    Normalized values from one Loc-Bench row.

Output:
    Immutable input, gold, metadata, and complete-instance models.

Pipeline stage:
    Dataset normalization and pipeline boundary.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    """Base configuration shared by normalized dataset models."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class GraphFixInput(StrictModel):
    """Information that localization inference is allowed to observe."""

    instance_id: str = Field(min_length=1)
    repo: str = Field(min_length=1)
    base_commit: str = Field(min_length=1)
    problem_statement: str = Field(min_length=1)
    category: str


class GraphFixGold(StrictModel):
    """Evaluation-only targets that must remain hidden during inference."""

    patch: str
    test_patch: str
    edit_functions: list[str]
    added_functions: list[str]
    labels: list[str] | None


class GraphFixMetadata(StrictModel):
    """Non-target benchmark metadata retained during normalization."""

    hints_text: str
    created_at: int
    edit_functions_length: int
    extra_fields: dict[str, Any] = Field(default_factory=dict)


class GraphFixInstance(StrictModel):
    """A normalized benchmark example with an explicit leakage boundary."""

    input: GraphFixInput
    gold: GraphFixGold
    metadata: GraphFixMetadata
