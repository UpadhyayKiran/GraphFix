"""
Purpose:
    Verify repositories behavior without requiring network access.

Input:
    Small temporary repositories, records, models, or fixtures created by the tests.

Output:
    Passing assertions when the implementation follows the Review-1 contract.

Pipeline stage:
    Regression testing for the GraphFix implementation.
"""

from __future__ import annotations

import subprocess
from collections.abc import Sequence
from pathlib import Path
from unittest.mock import patch

import pytest

from graphfix.datasets.models import GraphFixInput
from graphfix.repositories import (
    RepositoryManager,
    RepositoryRequest,
    RepositoryValidationError,
    github_repository_url,
)
from graphfix.repositories.git import GitCommandResult, GitOperationError, run_git


def git(repository: Path, *args: str) -> str:
    """Run Git only against a test-created local repository."""
    result = subprocess.run(
        ["git", *args],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
        shell=False,
        timeout=30,
    )
    return result.stdout.strip()


def create_source_repository(root: Path) -> tuple[Path, str, str]:
    """Create two inert text-only commits for historical checkout tests."""
    source = root / "source"
    source.mkdir()
    git(source, "init", "-b", "main")
    git(source, "config", "user.name", "GraphFix Test")
    git(source, "config", "user.email", "graphfix-test@example.invalid")
    state_file = source / "state.txt"
    state_file.write_text("before\n", encoding="utf-8")
    git(source, "add", "state.txt")
    git(source, "commit", "-m", "First state")
    first = git(source, "rev-parse", "HEAD")
    state_file.write_text("after\n", encoding="utf-8")
    git(source, "commit", "-am", "Second state")
    second = git(source, "rev-parse", "HEAD")
    return source, first, second


def manager_for(cache: Path, source: Path) -> RepositoryManager:
    """Create a manager whose remote is the local inert fixture repository."""
    return RepositoryManager(cache, url_builder=lambda _repo: str(source))


def test_github_repository_url_construction_and_validation() -> None:
    """Verify github repository url construction and validation."""
    assert github_repository_url("UXARRAY/uxarray") == "https://github.com/UXARRAY/uxarray.git"
    for invalid in ("missing-owner", "owner/repo/extra", "../repo", "owner/..", "owner/repo.git"):
        with pytest.raises(RepositoryValidationError):
            github_repository_url(invalid)


def test_reconstructs_requested_historical_commit_not_latest(tmp_path: Path) -> None:
    """Verify reconstructs requested historical commit not latest."""
    source, first, second = create_source_repository(tmp_path)
    request = RepositoryRequest(instance_id="fixture-1", repo="owner/repo", base_commit=first)

    checkout = manager_for(tmp_path / "cache", source).reconstruct(request)

    assert checkout.success
    assert checkout.resolved_commit == first
    assert checkout.resolved_commit != second
    assert checkout.checkout_path is not None
    checkout_path = Path(checkout.checkout_path)
    assert git(checkout_path, "rev-parse", "HEAD") == first
    assert (checkout_path / "state.txt").read_text(encoding="utf-8") == "before\n"


def test_nonexistent_commit_returns_explicit_failure(tmp_path: Path) -> None:
    """Verify nonexistent commit returns explicit failure."""
    source, _, _ = create_source_repository(tmp_path)
    request = RepositoryRequest(instance_id="missing", repo="owner/repo", base_commit="f" * 40)

    checkout = manager_for(tmp_path / "cache", source).reconstruct(request)

    assert not checkout.success
    assert checkout.failure is not None
    assert checkout.failure.category == "COMMIT_NOT_FOUND"
    assert checkout.resolved_commit is None


def test_cache_reuse_and_isolated_checkouts(tmp_path: Path) -> None:
    """Verify cache reuse and isolated checkouts."""
    source, first, second = create_source_repository(tmp_path)
    manager = manager_for(tmp_path / "cache", source)

    first_checkout = manager.reconstruct(
        RepositoryRequest(instance_id="instance-one", repo="owner/repo", base_commit=first)
    )
    second_checkout = manager.reconstruct(
        RepositoryRequest(instance_id="instance-two", repo="owner/repo", base_commit=second)
    )

    assert first_checkout.success and second_checkout.success
    assert not first_checkout.cache_reused
    assert second_checkout.cache_reused
    assert first_checkout.cache_repository_path == second_checkout.cache_repository_path
    assert first_checkout.checkout_path != second_checkout.checkout_path
    assert git(Path(first_checkout.checkout_path or ""), "rev-parse", "HEAD") == first
    assert git(Path(second_checkout.checkout_path or ""), "rev-parse", "HEAD") == second


def test_existing_checkout_is_reused_without_mutation(tmp_path: Path) -> None:
    """Verify existing checkout is reused without mutation."""
    source, first, _ = create_source_repository(tmp_path)
    manager = manager_for(tmp_path / "cache", source)
    request = RepositoryRequest(instance_id="same-instance", repo="owner/repo", base_commit=first)

    initial = manager.reconstruct(request)
    repeated = manager.reconstruct(request)

    assert initial.success and repeated.success
    assert repeated.cache_reused
    assert repeated.checkout_path == initial.checkout_path
    assert repeated.resolved_commit == first


@pytest.mark.parametrize("change", ["tracked", "untracked"])
def test_existing_checkout_with_local_changes_is_rejected(
    tmp_path: Path, change: str
) -> None:
    """Verify cached source cannot silently diverge from its recorded commit."""
    source, first, _second = create_source_repository(tmp_path)
    manager = manager_for(tmp_path / "cache", source)
    request = RepositoryRequest(instance_id="fixture-dirty", repo="owner/repo", base_commit=first)
    initial = manager.reconstruct(request)
    assert initial.success and initial.checkout_path is not None
    checkout = Path(initial.checkout_path)
    if change == "tracked":
        (checkout / "state.txt").write_text("locally changed\n", encoding="utf-8")
    else:
        (checkout / "untracked.txt").write_text("local file\n", encoding="utf-8")

    repeated = manager.reconstruct(request)

    assert not repeated.success
    assert repeated.failure is not None
    assert repeated.failure.category == "CHECKOUT_ERROR"
    assert "local modifications or untracked files" in repeated.failure.message


def test_request_is_constructed_only_from_graphfix_input() -> None:
    """Verify request is constructed only from graphfix input."""
    inference_input = GraphFixInput(
        instance_id="safe-input",
        repo="owner/repo",
        base_commit="a" * 40,
        problem_statement="Issue visible at inference time",
        category="Bug Report",
    )

    request = RepositoryRequest.from_graphfix_input(inference_input)

    assert request.model_dump() == {
        "instance_id": "safe-input",
        "repo": "owner/repo",
        "base_commit": "a" * 40,
    }
    assert set(RepositoryRequest.model_fields) == {"instance_id", "repo", "base_commit"}


def test_invalid_repository_is_a_failure_not_an_exception(tmp_path: Path) -> None:
    """Verify invalid repository is a failure not an exception."""
    request = RepositoryRequest(instance_id="bad-repo", repo="../unsafe", base_commit="a" * 40)

    checkout = RepositoryManager(tmp_path / "cache").reconstruct(request)

    assert not checkout.success
    assert checkout.failure is not None
    assert checkout.failure.category == "INVALID_REPOSITORY"


def test_generated_cache_paths_cannot_escape_root(tmp_path: Path) -> None:
    """Verify generated cache paths cannot escape root."""
    source, first, _ = create_source_repository(tmp_path)
    cache_root = tmp_path / "cache"
    request = RepositoryRequest(
        instance_id="../../outside",
        repo="owner/repo",
        base_commit=first,
    )

    checkout = manager_for(cache_root, source).reconstruct(request)

    assert checkout.success
    assert checkout.checkout_path is not None
    assert Path(checkout.checkout_path).resolve().is_relative_to(cache_root.resolve())
    assert ".." not in Path(checkout.checkout_path).parts


def test_checkout_uses_short_worktree_path_and_enables_long_paths(tmp_path: Path) -> None:
    """Verify checkout uses short worktree path and enables long paths."""
    source, first, _ = create_source_repository(tmp_path)
    cache_root = tmp_path / "cache"
    commands: list[tuple[str, ...]] = []

    def capturing_runner(
        args: Sequence[str],
        *,
        cwd: Path | None = None,
        timeout_seconds: float = 120.0,
    ) -> GitCommandResult:
        """Perform capturing runner."""
        commands.append(tuple(args))
        return run_git(args, cwd=cwd, timeout_seconds=timeout_seconds)

    manager = RepositoryManager(
        cache_root,
        url_builder=lambda _repo: str(source),
        git_runner=capturing_runner,
    )
    checkout = manager.reconstruct(
        RepositoryRequest(instance_id="long-path-test", repo="owner/repo", base_commit=first)
    )

    assert checkout.success
    assert checkout.checkout_path is not None
    relative = Path(checkout.checkout_path).relative_to(cache_root.resolve())
    assert relative.parts[0] == "w"
    assert len(relative.parts[2]) == 16
    worktree_command = next(
        command for command in commands if "worktree" in command and "add" in command
    )
    assert worktree_command[:2] == ("-c", "core.longpaths=true")


def test_timeout_is_categorized_without_crashing_batch(tmp_path: Path) -> None:
    """Verify timeout is categorized without crashing batch."""
    def timeout_runner(
        args: Sequence[str],
        *,
        cwd: Path | None = None,
        timeout_seconds: float = 120.0,
    ) -> GitCommandResult:
        """Perform timeout runner."""
        del args, cwd, timeout_seconds
        raise GitOperationError("TIMEOUT", "simulated timeout")

    manager = RepositoryManager(tmp_path / "cache", git_runner=timeout_runner)
    request = RepositoryRequest(instance_id="timeout", repo="owner/repo", base_commit="a" * 40)

    checkout = manager.reconstruct(request)

    assert not checkout.success
    assert checkout.failure is not None
    assert checkout.failure.category == "TIMEOUT"


def test_git_output_uses_explicit_resilient_utf8_decoding() -> None:
    """Verify git output uses explicit resilient utf8 decoding."""
    completed = subprocess.CompletedProcess(
        args=["git", "version"], returncode=0, stdout="git version fixture\n", stderr=""
    )
    with patch("graphfix.repositories.git.subprocess.run", return_value=completed) as mocked:
        result = run_git(["version"])

    assert result.stdout == "git version fixture"
    assert mocked.call_args.kwargs["encoding"] == "utf-8"
    assert mocked.call_args.kwargs["errors"] == "replace"
