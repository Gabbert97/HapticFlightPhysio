"""Shared physiological-processing utilities."""

from __future__ import annotations

import csv
import xml.etree.ElementTree as ET
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from multimodalphysiokit.core import Signal


def segment_signal(signal: Signal, start: int, end: int, label: str | None = None) -> Signal:
    if not 0 <= start < end <= signal.n_samples:
        raise ValueError(f"Invalid segment [{start}, {end}) for {signal.n_samples} samples")
    samples = signal.samples[start:end].copy()
    return Signal(samples=samples, time=np.arange(samples.size) / signal.sampling_frequency,
                  sampling_frequency=signal.sampling_frequency,
                  label=label or signal.label, units=signal.units,
                  metadata={**signal.metadata, "source_start_sample": start,
                            "source_end_sample": end})


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def participant_metadata_from_xlsx(path: Path) -> list[dict[str, str]]:
    """Read participant metadata using the workbook's named header columns."""
    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with ZipFile(path) as archive:
        shared_root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
        shared = ["".join(node.text or "" for node in item.iterfind(".//m:t", ns))
                  for item in shared_root.findall("m:si", ns)]
        sheet = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))
    rows = sheet.findall(".//m:sheetData/m:row", ns)
    if len(rows) < 2:
        raise ValueError(f"Workbook has no metadata header: {path}")

    def row_values(row):
        values = {}
        for cell in row.findall("m:c", ns):
            ref = cell.get("r", ""); column = "".join(ch for ch in ref if ch.isalpha())
            value_node = cell.find("m:v", ns); value = "" if value_node is None else value_node.text or ""
            if cell.get("t") == "s" and value:
                value = shared[int(value)]
            values[column] = value
        return values

    headers = {value.strip().casefold(): column for column, value in row_values(rows[1]).items()}
    required = {"subject id", "test group", "age"}
    if not required.issubset(headers):
        raise ValueError(f"Missing workbook columns: {sorted(required - set(headers))}")

    output = []
    seen: set[str] = set()
    for row in rows[2:]:
        values = row_values(row)
        subject_text = values.get(headers["subject id"], "")
        digits = "".join(ch for ch in subject_text if ch.isdigit())
        if not digits:
            continue
        participant_id = f"Subject_{int(digits)}"
        # Subject22 is an unused blank template row, not a dataset participant.
        group_raw = values.get(headers["test group"], "").strip().upper()
        age_raw = values.get(headers["age"], "").strip()
        if not group_raw and not age_raw:
            continue
        if participant_id in seen:
            raise ValueError(f"Duplicate participant in workbook: {participant_id}")
        seen.add(participant_id)
        group = {"HA": "Haptic", "NOHA": "NoHA"}.get(group_raw, group_raw)
        output.append({"participant_id": participant_id, "group": group,
                       "age_years": age_raw})
    return output
