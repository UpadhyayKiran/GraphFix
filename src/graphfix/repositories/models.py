"""
Purpose:
    Defines safe reconstruction requests and structured outcomes.

Input:
    GraphFixInput and repository operation details.

Output:
    Checkout, warning, and failure records.

Pipeline stage:
    Repository reconstruction contracts.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from graphfix.datasets.models import GraphFixInput

FailureCategory = Literal[
    "INVALID_REPOSITORY",
    "INVALID_COMMIT",
    "GIT_UNAVAILABLE",
    "CACHE_ERROR",
    "NETWORK_OR_FETCH_ERROR",
    "COMMIT_NOT_FOUND",
    "CHECKOUT_ERROR",
    "HEAD_MISMATCH",
    "TIMEOUT",
]


class StrictRepositoryModel(BaseModel):
    """Strict immutable configuration for repository-layer values."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class RepositoryRequest(StrictRepositoryModel):
    """Inference-safe information required to reconstruct one repository state."""

    instance_id: str = Field(min_length=1)
    repo: str = Field(min_length=1)
    base_commit: str = Field(min_length=1)

    @classmethod
    def from_graphfix_input(cls, value: GraphFixInput) -> RepositoryRequest:
        """Build a request solely from the inference-visible dataset model."""
        return cls(
            instance_id=value.instance_id,
            repo=value.repo,
            base_commit=value.base_commit,
        )


class RepositoryWarning(StrictRepositoryModel):
    """A non-fatal operational warning retained for audit."""

    code: str
    message: str


class RepositoryFailure(StrictRepositoryModel):
    """A categorized reconstruction failure for one benchmark instance."""

    category: FailureCategory
    message: str


class RepositoryCheckout(StrictRepositoryModel):
    """The verified outcome of one isolated historical checkout request."""

    instance_id: str
    repo: str
    requested_commit: str
    resolved_commit: str | None = None
    cache_repository_path: str | None = None
    checkout_path: str | None = None
    cache_reused: bool = False
    fetch_fallback_used: bool = False
    success: bool = False
    warnings: tuple[RepositoryWarning, ...] = ()
    failure: RepositoryFailure | None = None
    elapsed_seconds: float = Field(ge=0.0)
