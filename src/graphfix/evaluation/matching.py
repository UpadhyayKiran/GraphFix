"""
Purpose:
    Extracts normalized modification locations after localization.

Input:
    Evaluation-only GraphFixGold.

Output:
    Multi-file and file/function GoldLocations.

Pipeline stage:
    Gold extraction.
"""

from __future__ import annotations

import re
from pathlib import PurePosixPath

from pydantic import BaseModel, ConfigDict

from graphfix.datasets.models import GraphFixGold

_DIFF_HEADER = re.compile(r"^diff --git a/(.+?) b/(.+)$")


class GoldLocations(BaseModel):
    """Normalized evaluation-only file and function targets."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    files: tuple[str, ...]
    functions: tuple[str, ...]


def normalize_repository_path(value: str) -> str:
    """Normalize separators and common unified-diff prefixes without filesystem access."""
    cleaned = value.strip().replace("\\", "/")
    while cleaned.startswith("./"):
        cleaned = cleaned[2:]
    if cleaned.startswith(("a/", "b/")):
        cleaned = cleaned[2:]
    normalized = PurePosixPath(cleaned).as_posix()
    return normalized.casefold()


def normalize_function_target(value: str) -> str:
    """Normalize a Loc-Bench `path:function` target for exact matching."""
    path, separator, symbol = value.partition(":")
    if not separator:
        return value.strip().casefold()
    symbol = symbol.strip().removesuffix("()")
    return f"{normalize_repository_path(path)}:{symbol.casefold()}"


def extract_gold_locations(gold: GraphFixGold) -> GoldLocations:
    """Extract gold files from patch headers and callable targets without altering gold data."""
    files: set[str] = set()
    for line in gold.patch.splitlines():
        match = _DIFF_HEADER.match(line)
        if match:
            target = match.group(2)
            if target != "/dev/null":
                files.add(normalize_repository_path(target))
    functions = {
        normalize_function_target(value)
        for value in (*gold.edit_functions, *gold.added_functions)
        if value.strip()
    }
    files.update(target.split(":", 1)[0] for target in functions if ":" in target)
    return GoldLocations(files=tuple(sorted(files)), functions=tuple(sorted(functions)))
