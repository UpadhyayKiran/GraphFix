"""
Purpose:
    Runs bounded non-shell Git commands safely.

Input:
    Git arguments, working directory, and timeout.

Output:
    Captured UTF-8 output or categorized failure.

Pipeline stage:
    Low-level repository support.
"""

from __future__ import annotations

import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from graphfix.repositories.models import FailureCategory


@dataclass(frozen=True, slots=True)
class GitCommandResult:
    """Captured output from a successful Git command."""

    stdout: str
    stderr: str


class GitOperationError(RuntimeError):
    """A bounded Git command failure with a stable category."""

    def __init__(self, category: FailureCategory, message: str) -> None:
        """
        Purpose:
            Initialize the object and store the configuration needed by later operations.

        Args:
            category: The category value used by this operation.
            message: The message value used by this operation.

        Returns:
            Nothing. The operation updates state, writes output, or performs a check directly.

        Pipeline role:
            Exact historical repository reconstruction.
        """
        super().__init__(message)
        self.category = category


class GitRunner(Protocol):
    """Injectable Git command interface used by the repository manager."""

    def __call__(
        self,
        args: Sequence[str],
        *,
        cwd: Path | None = None,
        timeout_seconds: float = 120.0,
    ) -> GitCommandResult:
        """Run one bounded Git command and return its captured output."""
        ...


def run_git(
    args: Sequence[str],
    *,
    cwd: Path | None = None,
    timeout_seconds: float = 120.0,
) -> GitCommandResult:
    """Run Git with an argument list, captured output, and a finite timeout."""
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=cwd,
            check=False,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            text=True,
            shell=False,
            timeout=timeout_seconds,
        )
    except FileNotFoundError as error:
        raise GitOperationError("GIT_UNAVAILABLE", "Git executable was not found") from error
    except subprocess.TimeoutExpired as error:
        raise GitOperationError(
            "TIMEOUT", f"Git command exceeded {timeout_seconds:.1f} seconds"
        ) from error
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip().replace("\n", " ")[:500]
        raise GitOperationError(
            "NETWORK_OR_FETCH_ERROR",
            f"Git command failed with exit code {result.returncode}: {detail}",
        )
    return GitCommandResult(stdout=result.stdout.strip(), stderr=result.stderr.strip())
