"""Strict participant-level ECG polarity configuration helpers."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np


def _canonical_participant_id(participant_id: str | int) -> str:
    if isinstance(participant_id, int):
        return f"Subject_{participant_id}"
    value = str(participant_id).strip()
    if value.isdigit():
        return f"Subject_{int(value)}"
    return value


def read_ecg_polarity_config(path: str | Path) -> list[dict[str, str]]:
    """Read the authoritative ECG polarity configuration table."""
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"ECG polarity configuration not found: {config_path}")
    with config_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"participant_id", "ecg_polarity_final", "ecg_multiplier_final"}
    missing = required - set(rows[0] if rows else [])
    if missing:
        raise ValueError(f"ECG polarity configuration is missing columns: {sorted(missing)}")
    return rows


def get_ecg_multiplier(
    participant_id: str | int,
    config: str | Path | Iterable[Mapping[str, object]],
) -> int:
    """Return a finalized +1/-1 multiplier or raise an explicit error."""
    target = _canonical_participant_id(participant_id)
    rows = read_ecg_polarity_config(config) if isinstance(config, (str, Path)) else list(config)
    matches = [row for row in rows if str(row.get("participant_id", "")) == target]
    if not matches:
        raise KeyError(f"Participant is missing from ECG polarity configuration: {target}")
    if len(matches) != 1:
        raise ValueError(f"Duplicate ECG polarity configuration rows for {target}")
    value = str(matches[0].get("ecg_multiplier_final", "")).strip()
    try:
        multiplier = int(value)
    except ValueError as exc:
        raise ValueError(f"ECG polarity is unresolved for {target}") from exc
    if multiplier not in (-1, 1):
        raise ValueError(f"Invalid final ECG multiplier for {target}: {value!r}")
    return multiplier


def apply_ecg_polarity(
    ecg_values: object,
    participant_id: str | int,
    config: str | Path | Iterable[Mapping[str, object]],
) -> np.ndarray:
    """Return an in-memory polarity-corrected array; never write source data."""
    values = np.asarray(ecg_values)
    if not np.issubdtype(values.dtype, np.number):
        raise TypeError("ECG values must be numeric")
    multiplier = get_ecg_multiplier(participant_id, config)
    return values * multiplier
