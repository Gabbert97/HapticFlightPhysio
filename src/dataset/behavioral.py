"""Read behavioral trial boundaries without modifying behavioral files."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S.%f"


@dataclass(frozen=True)
class BehavioralBoundary:
    start: datetime | None
    end: datetime | None
    duration_s: float | None
    outcome: str
    status: str
    notes: str
    valid_timestamp_rows: int


def _parse_timestamp(value: str) -> datetime:
    # strptime accepts 1-6 fractional digits; the source data currently use 3.
    return datetime.strptime(value, TIMESTAMP_FORMAT)


def read_behavioral_boundary(path: Path) -> BehavioralBoundary:
    """Read first/last valid TimeStamp and an optional trailing outcome row."""
    if path.stat().st_size == 0:
        return BehavioralBoundary(None, None, None, "", "empty", "zero-byte file", 0)

    timestamps: list[datetime] = []
    outcome = ""
    invalid_timestamp_rows = 0

    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration:
            return BehavioralBoundary(None, None, None, "", "empty", "zero-row file", 0)

        try:
            timestamp_index = header.index("TimeStamp")
        except ValueError:
            return BehavioralBoundary(
                None, None, None, "", "empty", "header missing TimeStamp", 0
            )

        for row in reader:
            value = row[timestamp_index].strip() if len(row) > timestamp_index else ""
            if value:
                try:
                    timestamps.append(_parse_timestamp(value))
                except ValueError:
                    invalid_timestamp_rows += 1
                continue

            nonempty = [cell.strip() for cell in row if cell.strip()]
            if nonempty and nonempty[0].lower().startswith("trial "):
                outcome = ",".join(nonempty)
            elif nonempty:
                invalid_timestamp_rows += 1

    if not timestamps:
        notes = "header-only file"
        if invalid_timestamp_rows:
            notes += f"; {invalid_timestamp_rows} non-timestamp row(s)"
        return BehavioralBoundary(None, None, None, outcome, "empty", notes, 0)

    decreases = sum(current < previous for previous, current in zip(timestamps, timestamps[1:]))
    if decreases:
        return BehavioralBoundary(
            timestamps[0],
            timestamps[-1],
            (timestamps[-1] - timestamps[0]).total_seconds(),
            outcome,
            "invalid_timestamp",
            f"{decreases} backward timestamp transition(s)",
            len(timestamps),
        )

    duration = (timestamps[-1] - timestamps[0]).total_seconds()
    if duration <= 0:
        return BehavioralBoundary(
            timestamps[0], timestamps[-1], duration, outcome,
            "invalid_timestamp", "non-positive trial duration", len(timestamps)
        )

    notes = ""
    if invalid_timestamp_rows:
        notes = f"{invalid_timestamp_rows} unrecognized non-data row(s)"
    return BehavioralBoundary(
        timestamps[0], timestamps[-1], duration, outcome, "valid", notes, len(timestamps)
    )
