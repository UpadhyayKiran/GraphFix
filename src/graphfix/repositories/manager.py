"""
Purpose:
    Reconstructs the exact historical repository commit.

Input:
    An inference-safe RepositoryRequest.

Output:
    A verified RepositoryCheckout or categorized failure.

Pipeline stage:
    Repository reconstruction before parsing.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable
from pathlib import Path
from time import perf_counter
from typing import Final

from graphfix.repositories.git import GitOperationError, GitRunner, run_git
from graphfix.repositories.models import (
    FailureCategory,
    RepositoryCheckout,
    RepositoryFailure,
    RepositoryRequest,
    RepositoryWarning,
)

_REPOSITORY_PART: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_FULL_SHA: Final[re.Pattern[str]] = re.compile(r"^[0-9a-fA-F]{40}$")
_DIRECT_FETCH_DEPTH: Final[int] = 1
_FALLBACK_FETCH_DEPTH: Final[int] = 500


class RepositoryValidationError(ValueError):
    """Raised for an unsafe or malformed GitHub repository identifier."""


def github_repository_url(repository: str) -> str:
    """Validate owner/repository and return its canonical public Git URL."""
    parts = repository.split("/")
    if len(parts) != 2 or any(not part for part in parts):
        raise RepositoryValidationError("Repository must use owner/repository format")
    owner, name = parts
    if any(part in {".", ".."} or not _REPOSITORY_PART.fullmatch(part) for part in parts):
        raise RepositoryValidationError("Repository contains an unsafe owner or name")
    if name.lower().endswith(".git"):
        raise RepositoryValidationError("Repository name must not include a .git suffix")
    return f"https://github.com/{owner}/{name}.git"


class RepositoryManager:
    """Reconstruct exact commits using shared bare objects and isolated worktrees."""

    def __init__(
        self,
        cache_root: Path,
        *,
        url_builder: Callable[[str], str] = github_repository_url,
        git_runner: GitRunner = run_git,
        command_timeout_seconds: float = 120.0,
    ) -> None:
        """
        Purpose:
            Configure the safe cache, Git runner, URL builder, and command timeout
            used for checkout.

        Args:
            cache_root: Root folder for shared bare repositories and detached worktrees.
            url_builder: Function that validates a repository name and returns its clone URL.
            git_runner: Injectable bounded Git command runner; tests use a substitute runner.
            command_timeout_seconds: Maximum time allowed for each Git command.

        Returns:
            Nothing. The operation updates state, writes output, or performs a check directly.

        Raises:
            ValueError: If the command timeout is zero or negative.

        Pipeline role:
            Exact historical repository reconstruction.
        """
        if command_timeout_seconds <= 0:
            raise ValueError("command_timeout_seconds must be positive")
        self._cache_root = cache_root.resolve()
        self._url_builder = url_builder
        self._git = git_runner
        self._timeout = command_timeout_seconds

    @staticmethod
    def _key(value: str) -> str:
        """
        Purpose:
            Create a short stable cache key from a repository name.

        Args:
            value: Repository name or instance ID that needs a filesystem-safe key.

        Returns:
            The first 20 hexadecimal characters of the value's SHA-256 digest.

        Pipeline role:
            Exact historical repository reconstruction.
        """
        return hashlib.sha256(value.encode("utf-8")).hexdigest()[:20]

    def _paths(self, request: RepositoryRequest) -> tuple[Path, Path]:
        """
        Purpose:
            Build safe bare-cache and detached-worktree paths for one reconstruction request.

        Args:
            request: Validated repository reconstruction request.

        Returns:
            The bare-repository path and the short detached-worktree path.

        Raises:
            RepositoryValidationError: If either generated path would escape the cache root.

        Pipeline role:
            Exact historical repository reconstruction.
        """
        repo_key = self._key(request.repo)
        cache_repository = self._cache_root / "objects" / f"{repo_key}.git"
        checkout = (
            self._cache_root
            / "w"
            / repo_key
            / request.base_commit.lower()[:16]
            / self._key(request.instance_id)
        )
        for path in (cache_repository, checkout):
            if not path.resolve().is_relative_to(self._cache_root):
                raise RepositoryValidationError("Generated cache path escaped the cache root")
        return cache_repository, checkout

    def _run(self, args: list[str], *, cwd: Path | None = None) -> str:
        """
        Purpose:
            Run a Git command through the configured runner and return its completed process.

        Args:
            args: Git arguments without a shell command string.
            cwd: Optional working directory for the command.

        Returns:
            Trimmed standard output from the successful Git command.

        Pipeline role:
            Exact historical repository reconstruction.
        """
        result = self._git(args, cwd=cwd, timeout_seconds=self._timeout)
        return result.stdout

    def _has_commit(self, cache_repository: Path, commit: str) -> bool:
        """
        Purpose:
            Check whether the bare cache already contains the exact requested commit.

        Args:
            cache_repository: Bare Git repository to inspect.
            commit: Full commit SHA that must exist as a commit object.

        Returns:
            A `bool` result produced by the operation.

        Pipeline role:
            Exact historical repository reconstruction.
        """
        try:
            self._run(
                ["--git-dir", str(cache_repository), "cat-file", "-e", f"{commit}^{{commit}}"]
            )
        except GitOperationError:
            return False
        return True

    def _failure(
        self,
        request: RepositoryRequest,
        started: float,
        category: FailureCategory,
        message: str,
        *,
        cache_repository: Path | None = None,
        checkout: Path | None = None,
        cache_reused: bool = False,
        fallback_used: bool = False,
        warnings: tuple[RepositoryWarning, ...] = (),
    ) -> RepositoryCheckout:
        """
        Purpose:
            Build a failed checkout result without crashing a batch run.

        Args:
            request: Validated repository reconstruction request.
            started: The started value used by this operation.
            category: The category value used by this operation.
            message: The message value used by this operation.
            cache_repository: The cache repository value used by this operation.
            checkout: The checkout value used by this operation.
            cache_reused: The cache reused value used by this operation.
            fallback_used: The fallback used value used by this operation.
            warnings: The warnings value used by this operation.

        Returns:
            A `RepositoryCheckout` result produced by the operation.

        Pipeline role:
            Exact historical repository reconstruction.
        """
        return RepositoryCheckout(
            instance_id=request.instance_id,
            repo=request.repo,
            requested_commit=request.base_commit,
            cache_repository_path=str(cache_repository) if cache_repository else None,
            checkout_path=str(checkout) if checkout else None,
            cache_reused=cache_reused,
            fetch_fallback_used=fallback_used,
            success=False,
            warnings=warnings,
            failure=RepositoryFailure(category=category, message=message),
            elapsed_seconds=perf_counter() - started,
        )

    def reconstruct(self, request: RepositoryRequest) -> RepositoryCheckout:
        """Reconstruct one exact commit without executing any repository content."""
        started = perf_counter()
        try:
            remote_url = self._url_builder(request.repo)
        except (RepositoryValidationError, ValueError) as error:
            return self._failure(request, started, "INVALID_REPOSITORY", str(error))
        if not _FULL_SHA.fullmatch(request.base_commit):
            return self._failure(
                request, started, "INVALID_COMMIT", "base_commit must be a full 40-character SHA"
            )

        try:
            cache_repository, checkout = self._paths(request)
            cache_reused = cache_repository.exists()
            if cache_reused:
                if not cache_repository.is_dir():
                    return self._failure(
                        request,
                        started,
                        "CACHE_ERROR",
                        "Repository cache path is not a directory",
                        cache_repository=cache_repository,
                        checkout=checkout,
                        cache_reused=True,
                    )
                is_bare = self._run(
                    ["--git-dir", str(cache_repository), "rev-parse", "--is-bare-repository"]
                )
                if is_bare != "true":
                    return self._failure(
                        request,
                        started,
                        "CACHE_ERROR",
                        "Repository cache is not a bare Git repository",
                        cache_repository=cache_repository,
                        checkout=checkout,
                        cache_reused=True,
                    )
            else:
                cache_repository.parent.mkdir(parents=True, exist_ok=True)
                self._run(["init", "--bare", str(cache_repository)])
                self._run(
                    ["--git-dir", str(cache_repository), "remote", "add", "origin", remote_url]
                )
        except GitOperationError as error:
            cache_category: FailureCategory = (
                error.category
                if error.category in {"GIT_UNAVAILABLE", "TIMEOUT"}
                else "CACHE_ERROR"
            )
            return self._failure(
                request,
                started,
                cache_category,
                str(error),
            )
        except OSError as error:
            return self._failure(request, started, "CACHE_ERROR", str(error))

        warnings: list[RepositoryWarning] = []
        fallback_used = False
        if not self._has_commit(cache_repository, request.base_commit):
            try:
                self._run(
                    [
                        "--git-dir",
                        str(cache_repository),
                        "fetch",
                        "--no-tags",
                        f"--depth={_DIRECT_FETCH_DEPTH}",
                        "origin",
                        request.base_commit,
                    ]
                )
            except GitOperationError as direct_error:
                if direct_error.category in {"GIT_UNAVAILABLE", "TIMEOUT"}:
                    return self._failure(
                        request,
                        started,
                        direct_error.category,
                        str(direct_error),
                        cache_repository=cache_repository,
                        checkout=checkout,
                        cache_reused=cache_reused,
                    )
                fallback_used = True
                warnings.append(
                    RepositoryWarning(
                        code="DIRECT_FETCH_FAILED",
                        message=(
                            "Direct commit fetch failed; bounded branch-history fallback attempted"
                        ),
                    )
                )
                try:
                    self._run(
                        [
                            "--git-dir",
                            str(cache_repository),
                            "fetch",
                            "--no-tags",
                            f"--depth={_FALLBACK_FETCH_DEPTH}",
                            "origin",
                            "+refs/heads/*:refs/remotes/origin/*",
                        ]
                    )
                except GitOperationError as fallback_error:
                    fetch_category: FailureCategory = (
                        fallback_error.category
                        if fallback_error.category in {"GIT_UNAVAILABLE", "TIMEOUT"}
                        else "NETWORK_OR_FETCH_ERROR"
                    )
                    return self._failure(
                        request,
                        started,
                        fetch_category,
                        str(fallback_error),
                        cache_repository=cache_repository,
                        checkout=checkout,
                        cache_reused=cache_reused,
                        fallback_used=True,
                        warnings=tuple(warnings),
                    )

        if not self._has_commit(cache_repository, request.base_commit):
            return self._failure(
                request,
                started,
                "COMMIT_NOT_FOUND",
                "Requested base_commit is unavailable after bounded fetch attempts",
                cache_repository=cache_repository,
                checkout=checkout,
                cache_reused=cache_reused,
                fallback_used=fallback_used,
                warnings=tuple(warnings),
            )

        try:
            if checkout.exists():
                resolved = self._run(["-C", str(checkout), "rev-parse", "HEAD"])
            else:
                checkout.parent.mkdir(parents=True, exist_ok=True)
                self._run(["--git-dir", str(cache_repository), "worktree", "prune"])
                self._run(
                    [
                        "-c",
                        "core.longpaths=true",
                        "--git-dir",
                        str(cache_repository),
                        "worktree",
                        "add",
                        "--detach",
                        str(checkout),
                        request.base_commit,
                    ]
                )
                resolved = self._run(["-C", str(checkout), "rev-parse", "HEAD"])
        except GitOperationError as error:
            checkout_category: FailureCategory = (
                error.category
                if error.category in {"GIT_UNAVAILABLE", "TIMEOUT"}
                else "CHECKOUT_ERROR"
            )
            return self._failure(
                request,
                started,
                checkout_category,
                str(error),
                cache_repository=cache_repository,
                checkout=checkout,
                cache_reused=cache_reused,
                fallback_used=fallback_used,
                warnings=tuple(warnings),
            )

        if resolved.lower() != request.base_commit.lower():
            return RepositoryCheckout(
                instance_id=request.instance_id,
                repo=request.repo,
                requested_commit=request.base_commit,
                resolved_commit=resolved,
                cache_repository_path=str(cache_repository),
                checkout_path=str(checkout),
                cache_reused=cache_reused,
                fetch_fallback_used=fallback_used,
                success=False,
                warnings=tuple(warnings),
                failure=RepositoryFailure(
                    category="HEAD_MISMATCH",
                    message="Resolved HEAD does not equal requested base_commit",
                ),
                elapsed_seconds=perf_counter() - started,
            )
        return RepositoryCheckout(
            instance_id=request.instance_id,
            repo=request.repo,
            requested_commit=request.base_commit,
            resolved_commit=resolved,
            cache_repository_path=str(cache_repository),
            checkout_path=str(checkout),
            cache_reused=cache_reused,
            fetch_fallback_used=fallback_used,
            success=True,
            warnings=tuple(warnings),
            elapsed_seconds=perf_counter() - started,
        )
