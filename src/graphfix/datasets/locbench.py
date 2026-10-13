"""
Purpose:
    Loads official Loc-Bench V1 and coordinates normalization.

Input:
    A provider or dataset-like row collection.

Output:
    A validated 560-row LocBenchDataset.

Pipeline stage:
    First pipeline stage.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final, Protocol, cast

from graphfix.datasets.models import GraphFixInput, GraphFixInstance
from graphfix.datasets.validation import (
    DatasetAnomaly,
    LocBenchValidationError,
    normalize_locbench_record,
    validate_raw_schema,
)

DEFAULT_DATASET_NAME: Final[str] = "czlll/Loc-Bench_V1"
DEFAULT_SPLIT: Final[str] = "test"


class RawDataset(Protocol):
    """Small structural interface used from a Hugging Face dataset."""

    @property
    def column_names(self) -> Sequence[str]:
        """Return the field names supplied by the raw dataset."""
        ...

    def __iter__(self) -> Iterator[Mapping[str, Any]]:
        """Yield raw Loc-Bench rows for validation and normalization."""
        ...


DatasetProvider = Callable[[str, str], RawDataset]


class LocBenchLoadError(RuntimeError):
    """Raised when the external dataset cannot be loaded or normalized."""


@dataclass(frozen=True, slots=True)
class LocBenchDataset:
    """Normalized records and warnings from one dataset split."""

    dataset_name: str
    split: str
    raw_fields: tuple[str, ...]
    instances: tuple[GraphFixInstance, ...]
    anomalies: tuple[DatasetAnomaly, ...]

    def inference_inputs(self) -> Iterator[GraphFixInput]:
        """Yield only fields that are permitted during localization inference."""
        return (instance.input for instance in self.instances)

    def __len__(self) -> int:
        """
        Purpose:
            Return the number of stored records.

        Returns:
            A `int` result produced by the operation.

        Pipeline role:
            Dataset preparation, before any repository work.
        """
        return len(self.instances)


def _huggingface_provider(dataset_name: str, split: str) -> RawDataset:
    """
    Purpose:
        Perform huggingface provider.

    Args:
        dataset_name: The dataset name value used by this operation.
        split: The split value used by this operation.

    Returns:
        A `RawDataset` result produced by the operation.

    Pipeline role:
        Dataset preparation, before any repository work.
    """
    from datasets import load_dataset  # type: ignore[import-untyped]

    dataset = load_dataset(dataset_name, split=split)
    return cast(RawDataset, dataset)


def normalize_locbench_records(
    records: Iterable[Mapping[str, Any]],
    *,
    raw_fields: Sequence[str],
    dataset_name: str = DEFAULT_DATASET_NAME,
    split: str = DEFAULT_SPLIT,
    limit: int | None = None,
) -> LocBenchDataset:
    """Normalize injected records; useful for tests and offline data sources."""
    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative or None")
    observed_fields = validate_raw_schema(raw_fields)
    instances: list[GraphFixInstance] = []
    anomalies: list[DatasetAnomaly] = []
    for index, record in enumerate(records):
        if limit is not None and index >= limit:
            break
        try:
            instance, row_anomalies = normalize_locbench_record(record)
        except LocBenchValidationError as error:
            identifier = record.get("instance_id", f"row {index}")
            raise LocBenchValidationError(f"Failed to normalize {identifier!r}: {error}") from error
        instances.append(instance)
        anomalies.extend(row_anomalies)
    return LocBenchDataset(
        dataset_name=dataset_name,
        split=split,
        raw_fields=observed_fields,
        instances=tuple(instances),
        anomalies=tuple(anomalies),
    )


def load_locbench(
    dataset_name: str = DEFAULT_DATASET_NAME,
    split: str = DEFAULT_SPLIT,
    *,
    limit: int | None = None,
    provider: DatasetProvider | None = None,
) -> LocBenchDataset:
    """Load and normalize Loc-Bench from Hugging Face or an injected provider."""
    selected_provider = provider or _huggingface_provider
    try:
        raw_dataset = selected_provider(dataset_name, split)
    except Exception as error:
        raise LocBenchLoadError(
            f"Unable to load dataset '{dataset_name}' split '{split}': {error}"
        ) from error
    try:
        return normalize_locbench_records(
            raw_dataset,
            raw_fields=raw_dataset.column_names,
            dataset_name=dataset_name,
            split=split,
            limit=limit,
        )
    except (LocBenchValidationError, ValueError) as error:
        raise LocBenchLoadError(
            f"Unable to normalize dataset '{dataset_name}' split '{split}': {error}"
        ) from error
