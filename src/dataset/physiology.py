"""Read timing metadata from biosignalsplux HDF5 recordings."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import h5py


@dataclass(frozen=True)
class PhysiologyTiming:
    start: datetime
    end: datetime
    sampling_rate_hz: float
    n_samples: int
    duration_s: float
    device_group: str
    status: str
    notes: str


def _parse_hdf5_datetime(date_value: object, time_value: object) -> datetime:
    date_text = str(date_value)
    time_text = str(time_value)
    date_part = datetime.strptime(date_text, "%Y-%m-%d").date()
    time_part = datetime.strptime(time_text, "%H:%M:%S.%f").time()
    return datetime.combine(date_part, time_part)


def read_physiology_timing(path: Path, expected_rate_hz: float = 1000.0) -> PhysiologyTiming:
    """Read only HDF5 structure/metadata; never load raw signal arrays."""
    with h5py.File(path, "r") as handle:
        device_groups = [
            name
            for name, value in handle.items()
            if isinstance(value, h5py.Group)
            and {"date", "time", "sampling rate", "nsamples"}.issubset(value.attrs)
        ]
        if len(device_groups) != 1:
            raise ValueError(
                f"Expected one biosignalsplux device group in {path}, found {device_groups}"
            )

        device_group = device_groups[0]
        attrs = handle[device_group].attrs
        start = _parse_hdf5_datetime(attrs["date"], attrs["time"])
        sampling_rate = float(attrs["sampling rate"])
        n_samples = int(attrs["nsamples"])

    if sampling_rate <= 0:
        raise ValueError(f"Non-positive sampling rate in {path}: {sampling_rate}")
    if n_samples <= 0:
        raise ValueError(f"Non-positive nsamples in {path}: {n_samples}")

    duration_s = n_samples / sampling_rate
    end = start + timedelta(seconds=duration_s)
    if end <= start:
        raise ValueError(f"Non-positive physiological interval in {path}")

    notes = ""
    status = "valid"
    if sampling_rate != expected_rate_hz:
        status = "warning"
        notes = f"unexpected sampling rate: {sampling_rate:g} Hz"

    return PhysiologyTiming(
        start, end, sampling_rate, n_samples, duration_s, device_group, status, notes
    )
