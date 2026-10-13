"""
Purpose:
    Reconstruct a bounded set of Loc-Bench repositories at their exact base commits.

Input:
    Loc-Bench rows, cache directory, optional limit, and optional instance IDs.

Output:
    Verified detached worktrees and a reconstruction audit JSON file.

Pipeline stage:
    Repository preparation after dataset loading.

When to run:
    Run this script when you need this operational check or stage.

Run example:
    python scripts/reconstruct_locbench_repos.py --help
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from graphfix.datasets import DEFAULT_DATASET_NAME, DEFAULT_SPLIT, load_locbench
from graphfix.repositories import RepositoryCheckout, RepositoryManager, RepositoryRequest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE_ROOT = PROJECT_ROOT / "data" / "cache" / "repositories"
AUDIT_PATH = PROJECT_ROOT / "results" / "generated" / "repository_reconstruction_audit.json"


def parse_args() -> argparse.Namespace:
    """Parse the bounded development reconstruction limit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=3, help="Number of first inputs to attempt")
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit must be non-negative")
    return args


def _audit_record(checkout: RepositoryCheckout) -> dict[str, Any]:
    """
    Purpose:
        Perform audit record.

    Args:
        checkout: The checkout value used by this operation.

    Returns:
        A `dict[str, Any]` result produced by the operation.

    Pipeline role:
        Operational script support.
    """
    failure = checkout.failure
    return {
        "instance_id": checkout.instance_id,
        "repo": checkout.repo,
        "requested_commit": checkout.requested_commit,
        "resolved_commit": checkout.resolved_commit,
        "success": checkout.success,
        "cache_reused": checkout.cache_reused,
        "fetch_fallback_used": checkout.fetch_fallback_used,
        "elapsed_seconds": checkout.elapsed_seconds,
        "failure_category": failure.category if failure else None,
        "failure_message": failure.message if failure else None,
        "checkout_path": checkout.checkout_path,
        "warning_codes": [warning.code for warning in checkout.warnings],
    }


def main() -> None:
    """Run a safe reconstruction batch and write its compact ignored audit."""
    args = parse_args()
    dataset = load_locbench(limit=args.limit)
    manager = RepositoryManager(DEFAULT_CACHE_ROOT)
    checkouts = []
    for graphfix_input in dataset.inference_inputs():
        request = RepositoryRequest.from_graphfix_input(graphfix_input)
        checkout = manager.reconstruct(request)
        checkouts.append(checkout)
        requested = checkout.requested_commit[:12]
        resolved = checkout.resolved_commit[:12] if checkout.resolved_commit else "-"
        failure_category = checkout.failure.category if checkout.failure else "UNKNOWN"
        status = "SUCCESS" if checkout.success else f"FAILED:{failure_category}"
        cache = "reused" if checkout.cache_reused else "created"
        print(f"Instance: {checkout.instance_id}")
        print(f"  Repository: {checkout.repo}")
        print(f"  Requested: {requested}")
        print(f"  Resolved: {resolved}")
        print(f"  Status: {status}")
        print(f"  Cache: {cache}")
        print(f"  Fetch fallback: {checkout.fetch_fallback_used}")
        print(f"  Elapsed: {checkout.elapsed_seconds:.2f}s")
        print(f"  Checkout: {checkout.checkout_path or '-'}")

    audit = {
        "dataset": DEFAULT_DATASET_NAME,
        "split": DEFAULT_SPLIT,
        "attempted": len(checkouts),
        "successful": sum(checkout.success for checkout in checkouts),
        "failed": sum(not checkout.success for checkout in checkouts),
        "records": [_audit_record(checkout) for checkout in checkouts],
    }
    AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_PATH.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(
        f"Summary: attempted={audit['attempted']} successful={audit['successful']} "
        f"failed={audit['failed']}"
    )
    print(f"Audit: {AUDIT_PATH}")


if __name__ == "__main__":
    main()
