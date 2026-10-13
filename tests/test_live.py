"""Tests for generic, pinned, gold-free live repository localization."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from graphfix.live import (
    LiveLocalizationService,
    live_input,
    parse_public_github_url,
    resolve_default_branch_head,
    save_live_artifact,
)
from graphfix.repositories.git import GitCommandResult
from graphfix.repositories.models import RepositoryCheckout

SHA = "a" * 40


class FakeManager:
    def __init__(self, checkout: Path) -> None:
        self.checkout = checkout
        self.requests: list[Any] = []

    def reconstruct(self, request: Any) -> RepositoryCheckout:
        self.requests.append(request)
        return RepositoryCheckout(
            instance_id=request.instance_id,
            repo=request.repo,
            requested_commit=request.base_commit,
            resolved_commit=request.base_commit,
            checkout_path=str(self.checkout),
            cache_repository_path=str(self.checkout / ".git"),
            cache_reused=True,
            success=True,
            elapsed_seconds=0.01,
        )


def test_url_validation_and_default_head_pinning() -> None:
    def fake_git(
        args: Sequence[str],
        *,
        cwd: Path | None = None,
        timeout_seconds: float = 120.0,
    ) -> GitCommandResult:
        del cwd, timeout_seconds
        assert args[:2] == ["ls-remote", "--symref"]
        return GitCommandResult(
            stdout=f"ref: refs/heads/main\tHEAD\n{SHA}\tHEAD", stderr=""
        )

    assert parse_public_github_url("https://github.com/example/project.git") == "example/project"
    assert resolve_default_branch_head(
        "https://github.com/example/project", git_runner=fake_git
    ) == ("example/project", SHA, "main")
    with pytest.raises(ValueError):
        parse_public_github_url("https://gitlab.com/example/project")
    with pytest.raises(ValueError):
        parse_public_github_url("https://github.com/example/project/issues/1")


def test_generic_live_input_preparation_ranking_and_separate_storage(tmp_path: Path) -> None:
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    (checkout / "alpha.py").write_text(
        "def parse_scientific(value: str) -> float:\n    return float(value)\n",
        encoding="utf-8",
    )
    issue = live_input(
        "example/project", SHA, "Scientific notation", "Parse exponent numeric answers"
    )
    assert "gold" not in type(issue).model_fields
    manager = FakeManager(checkout)
    service = LiveLocalizationService(
        tmp_path / "cache", repository_manager=manager  # type: ignore[arg-type]
    )
    prepared = service.prepare(issue)
    result = service.bm25(prepared)
    assert prepared.issue.base_commit == SHA
    assert result.file_rankings[0].file_path == "alpha.py"
    assert result.function_rankings[0].symbol_name == "parse_scientific"
    output = tmp_path / "live-results"
    artifact = save_live_artifact(
        output,
        "https://github.com/example/project",
        "main",
        prepared,
        {"bm25": result},
    )
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    assert artifact.is_relative_to(output)
    assert payload["resolved_commit"] == SHA
    assert payload["gold"] is None and payload["metrics"] is None
    assert payload["methods"]["bm25"]["file_rankings"]
