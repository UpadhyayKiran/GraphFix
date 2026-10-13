"""
Purpose:
    Download, validate, and save a reproducible local Loc-Bench snapshot.

Input:
    The official dataset split, cache directory, output directory, and overwrite option.

Output:
    A native snapshot, full JSONL, safe browse CSV, and checksum manifest.

Pipeline stage:
    Dataset materialization before offline experiments.

When to run:
    Run this script when you need this operational check or stage.

Run example:
    python scripts/materialize_locbench.py --help
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, Protocol, cast

DATASET_ID: Final[str] = "czlll/Loc-Bench_V1"
SPLIT: Final[str] = "test"
EXPECTED_ROWS: Final[int] = 560
FULL_EXPORT_NAME: Final[str] = "Loc-Bench_V1_FULL.jsonl"
BROWSE_EXPORT_NAME: Final[str] = "Loc-Bench_V1_BROWSE.csv"
BROWSE_EXCLUDED_COLUMNS: Final[frozenset[str]] = frozenset({"patch", "test_patch"})
REQUIRED_COLUMNS: Final[frozenset[str]] = frozenset(
    {
        "repo",
        "instance_id",
        "base_commit",
        "patch",
        "test_patch",
        "problem_statement",
        "category",
        "edit_functions",
        "added_functions",
        "labels",
    }
)
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
RAW_DATA_ROOT: Final[Path] = PROJECT_ROOT / "data" / "raw"
DEFAULT_OUTPUT_DIR: Final[Path] = RAW_DATA_ROOT / "locbench_v1"
DEFAULT_CACHE_DIR: Final[Path] = Path.home() / ".cache" / "huggingface" / "datasets"


class MaterializableDataset(Protocol):
    """Dataset operations required without coupling tests to network loading."""

    @property
    def column_names(self) -> Sequence[str]:
        """Return the available raw dataset field names."""
        ...

    def __len__(self) -> int:
        """Return the number of raw rows."""
        ...

    def __iter__(self) -> Iterator[Mapping[str, Any]]:
        """Yield raw rows for validation and export."""
        ...

    def save_to_disk(self, dataset_path: str) -> None:
        """Save the provider's native snapshot at the requested path."""
        ...


DatasetReloader = Callable[[Path], MaterializableDataset]


@dataclass(frozen=True, slots=True)
class MaterializationResult:
    """Verified paths and metadata for one completed materialization."""

    dataset_id: str
    split: str
    row_count: int
    column_names: tuple[str, ...]
    snapshot_path: Path
    full_jsonl_path: Path
    browse_csv_path: Path
    manifest_path: Path
    full_jsonl_sha256: str
    browse_csv_sha256: str


def _reload_dataset(path: Path) -> MaterializableDataset:
    """
    Purpose:
        Perform reload dataset.

    Args:
        path: The file or directory path to use.

    Returns:
        A `MaterializableDataset` result produced by the operation.

    Pipeline role:
        Operational script support.
    """
    from datasets import load_from_disk  # type: ignore[import-untyped]

    return cast(MaterializableDataset, load_from_disk(str(path)))


def _load_dataset(dataset_id: str, split: str, cache_dir: Path) -> MaterializableDataset:
    """
    Purpose:
        Perform load dataset.

    Args:
        dataset_id: The dataset id value used by this operation.
        split: The split value used by this operation.
        cache_dir: The cache dir value used by this operation.

    Returns:
        A `MaterializableDataset` result produced by the operation.

    Pipeline role:
        Operational script support.
    """
    from datasets import load_dataset

    return cast(
        MaterializableDataset,
        load_dataset(dataset_id, split=split, cache_dir=str(cache_dir)),
    )


def resolve_output_dir(output_dir: Path, *, raw_data_root: Path = RAW_DATA_ROOT) -> Path:
    """Resolve and constrain outputs to a child of the project raw-data directory."""
    resolved_root = raw_data_root.resolve()
    resolved_output = output_dir.resolve()
    if resolved_output == resolved_root or not resolved_output.is_relative_to(resolved_root):
        raise ValueError(f"Output directory must be a child of '{resolved_root}'")
    return resolved_output


def validate_dataset(
    dataset: MaterializableDataset,
    *,
    expected_rows: int,
) -> tuple[str, ...]:
    """Validate the source row count and required raw schema without rewriting it."""
    row_count = len(dataset)
    if row_count != expected_rows:
        raise ValueError(f"Expected {expected_rows} rows, found {row_count}")
    columns = tuple(dataset.column_names)
    if len(columns) != len(set(columns)):
        raise ValueError("Dataset contains duplicate column names")
    missing = sorted(REQUIRED_COLUMNS.difference(columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing)}")
    return columns


def _sha256(path: Path) -> str:
    """
    Purpose:
        Calculate the SHA-256 checksum of a file.

    Args:
        path: The file or directory path to use.

    Returns:
        A `str` result produced by the operation.

    Pipeline role:
        Operational script support.
    """
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _csv_value(value: Any) -> str | int | float | bool:
    """
    Purpose:
        Convert a dataset value to the stable text form used in the browse CSV.

    Args:
        value: The value to check or convert.

    Returns:
        A `str | int | float | bool` result produced by the operation.

    Raises:
        TypeError: If the input or operation cannot be completed safely.

    Pipeline role:
        Operational script support.
    """
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    if isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Unsupported CSV value type: {type(value).__name__}")


def export_full_jsonl(
    dataset: MaterializableDataset,
    path: Path,
    *,
    columns: Sequence[str],
) -> int:
    """Export all rows and original columns as newline-delimited JSON."""
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in dataset:
            if tuple(row) != tuple(columns):
                raise ValueError("A dataset row does not match the declared column order")
            stream.write(json.dumps(dict(row), ensure_ascii=False, separators=(",", ":")))
            stream.write("\n")
            count += 1
    return count


def export_browse_csv(
    dataset: MaterializableDataset,
    path: Path,
    *,
    columns: Sequence[str],
) -> int:
    """Export every row while omitting only raw patch bodies for easier browsing."""
    browse_columns = tuple(column for column in columns if column not in BROWSE_EXCLUDED_COLUMNS)
    count = 0
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=browse_columns, lineterminator="\n")
        writer.writeheader()
        for row in dataset:
            writer.writerow({column: _csv_value(row[column]) for column in browse_columns})
            count += 1
    return count


def _remove_existing_raw_outputs(paths: Sequence[Path]) -> None:
    """
    Purpose:
        Remove only known generated dataset outputs before an explicit overwrite.

    Args:
        paths: The paths value used by this operation.

    Returns:
        Nothing. The operation updates state, writes output, or performs a check directly.

    Raises:
        ValueError: If the input or operation cannot be completed safely.

    Pipeline role:
        Operational script support.
    """
    for path in paths:
        if not path.exists():
            continue
        if path.is_symlink():
            raise ValueError(f"Refusing to replace symbolic link: {path}")
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def materialize_dataset(
    dataset: MaterializableDataset,
    *,
    output_dir: Path,
    cache_dir: Path,
    dataset_id: str = DATASET_ID,
    split: str = SPLIT,
    expected_rows: int = EXPECTED_ROWS,
    overwrite: bool = False,
    reloader: DatasetReloader = _reload_dataset,
    materialized_at: datetime | None = None,
    raw_data_root: Path = RAW_DATA_ROOT,
) -> MaterializationResult:
    """Save, export, reload, checksum, and describe an injected raw dataset."""
    resolved_output = resolve_output_dir(output_dir, raw_data_root=raw_data_root)
    columns = validate_dataset(dataset, expected_rows=expected_rows)
    snapshot_path = resolved_output / "dataset"
    full_path = resolved_output / FULL_EXPORT_NAME
    browse_path = resolved_output / BROWSE_EXPORT_NAME
    manifest_path = resolved_output / "manifest.json"
    raw_outputs = (snapshot_path, full_path, browse_path)
    existing = [path for path in raw_outputs if path.exists()]
    if existing and not overwrite:
        joined = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"Raw outputs already exist; use --overwrite: {joined}")
    if overwrite:
        _remove_existing_raw_outputs(raw_outputs)
    resolved_output.mkdir(parents=True, exist_ok=True)

    dataset.save_to_disk(str(snapshot_path))
    reloaded = reloader(snapshot_path)
    reloaded_columns = validate_dataset(reloaded, expected_rows=expected_rows)
    if reloaded_columns != columns:
        raise ValueError("Reloaded snapshot column order differs from the source dataset")

    full_count = export_full_jsonl(dataset, full_path, columns=columns)
    browse_count = export_browse_csv(dataset, browse_path, columns=columns)
    if full_count != expected_rows or browse_count != expected_rows:
        raise ValueError(
            f"Export row-count mismatch: JSONL={full_count}, CSV={browse_count}, "
            f"expected={expected_rows}"
        )

    full_checksum = _sha256(full_path)
    browse_checksum = _sha256(browse_path)
    timestamp = materialized_at or datetime.now(UTC)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "dataset_id": dataset_id,
        "split": split,
        "row_count": expected_rows,
        "column_names": list(columns),
        "materialized_at": timestamp.isoformat(),
        "source": {
            "provider": "Hugging Face",
            "cache_dir": str(cache_dir.resolve()),
        },
        "canonical_snapshot": {
            "format": "Hugging Face Dataset.save_to_disk",
            "path": "dataset",
        },
        "exports": {
            "full_jsonl": {
                "filename": FULL_EXPORT_NAME,
                "size_bytes": full_path.stat().st_size,
                "sha256": full_checksum,
                "columns": "all original columns",
            },
            "browse_csv": {
                "filename": BROWSE_EXPORT_NAME,
                "size_bytes": browse_path.stat().st_size,
                "sha256": browse_checksum,
                "excluded_columns": sorted(BROWSE_EXCLUDED_COLUMNS),
            },
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return MaterializationResult(
        dataset_id=dataset_id,
        split=split,
        row_count=expected_rows,
        column_names=columns,
        snapshot_path=snapshot_path,
        full_jsonl_path=full_path,
        browse_csv_path=browse_path,
        manifest_path=manifest_path,
        full_jsonl_sha256=full_checksum,
        browse_csv_sha256=browse_checksum,
    )


def parse_args() -> argparse.Namespace:
    """Parse safe materialization options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", default=DATASET_ID)
    parser.add_argument("--split", choices=(SPLIT,), default=SPLIT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Load the official split and materialize verified local raw outputs."""
    args = parse_args()
    output_dir = resolve_output_dir(args.output_dir)
    print(f"Loading dataset: {args.dataset_id}")
    print(f"Split: {args.split}")
    print(f"Hugging Face cache: {args.cache_dir.resolve()}")
    dataset = _load_dataset(args.dataset_id, args.split, args.cache_dir)
    result = materialize_dataset(
        dataset,
        output_dir=output_dir,
        cache_dir=args.cache_dir,
        dataset_id=args.dataset_id,
        split=args.split,
        overwrite=args.overwrite,
    )
    print(f"Rows: {result.row_count}")
    print(f"Columns ({len(result.column_names)}): {', '.join(result.column_names)}")
    print(f"Canonical snapshot: {result.snapshot_path}")
    print(f"FULL JSONL: {result.full_jsonl_path}")
    print(f"FULL JSONL SHA-256: {result.full_jsonl_sha256}")
    print(f"BROWSE CSV: {result.browse_csv_path}")
    print(f"BROWSE CSV SHA-256: {result.browse_csv_sha256}")
    print(f"Manifest: {result.manifest_path}")


if __name__ == "__main__":
    main()
