"""Extract observed digital rising-edge markers from baseline HDF5 files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np


DIGITAL_DATASET = "digital/digital_1"
EVENT_TABLE_DATASET = "events/digital"


@dataclass(frozen=True)
class BaselineDigitalEvent:
    event_number: str
    event_sample: int
    event_time_s: float
    digital_previous_value: int
    digital_new_value: int
    source_dataset: str


def _recording_group(handle: h5py.File) -> tuple[str, h5py.Group]:
    groups = [
        (name, item)
        for name, item in handle.items()
        if isinstance(item, h5py.Group)
        and "sampling rate" in item.attrs
        and "nsamples" in item.attrs
    ]
    if len(groups) != 1:
        raise ValueError(f"Expected one physiological device group, found {len(groups)}")
    return groups[0]


def read_baseline_rising_edges(path: Path) -> tuple[list[BaselineDigitalEvent], float, int]:
    """Return all observed 0 -> 1 transitions and verify the HDF5 event table.

    Events are point markers only. This function does not infer intervals or
    assign protocol meaning to the digital state.
    """
    with h5py.File(path, "r") as handle:
        group_name, group = _recording_group(handle)
        sampling_rate = float(group.attrs["sampling rate"])
        n_samples = int(group.attrs["nsamples"])
        if not np.isfinite(sampling_rate) or sampling_rate <= 0:
            raise ValueError(f"Invalid sampling rate in {path}: {sampling_rate}")

        if DIGITAL_DATASET not in group:
            raise KeyError(f"Missing /{group_name}/{DIGITAL_DATASET} in {path}")
        digital = np.asarray(group[DIGITAL_DATASET][()]).reshape(-1)
        if digital.size != n_samples:
            raise ValueError(
                f"Digital sample count differs from nsamples in {path}: "
                f"{digital.size} != {n_samples}"
            )
        rising = np.flatnonzero((digital[:-1] == 0) & (digital[1:] == 1)) + 1

        if EVENT_TABLE_DATASET not in group:
            raise KeyError(f"Missing /{group_name}/{EVENT_TABLE_DATASET} in {path}")
        table = np.asarray(group[EVENT_TABLE_DATASET][()])
        if table.ndim != 2 or table.shape[1] < 3:
            raise ValueError(f"Unexpected event-table shape in {path}: {table.shape}")
        table_rising = table[table[:, 2] == 1, 1].astype(np.int64)
        if not np.array_equal(rising.astype(np.int64), table_rising):
            raise ValueError(
                f"Rising-edge disagreement in {path}: digital_1={rising.tolist()}, "
                f"events/digital={table_rising.tolist()}"
            )

    source_dataset = f"/{group_name}/{DIGITAL_DATASET}"
    events = [
        BaselineDigitalEvent(
            event_number=f"E{number}",
            event_sample=int(sample),
            event_time_s=float(sample) / sampling_rate,
            digital_previous_value=0,
            digital_new_value=1,
            source_dataset=source_dataset,
        )
        for number, sample in enumerate(rising, start=1)
    ]
    return events, sampling_rate, n_samples
