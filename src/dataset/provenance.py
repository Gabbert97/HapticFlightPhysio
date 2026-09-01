"""Append-only provenance helpers for dataset preparation operations."""

from __future__ import annotations

import csv
import os
from datetime import datetime, timezone
from pathlib import Path

PROCESSING_LOG_FIELDS = ["timestamp", "operation", "script", "target", "status", "details"]


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def append_csv(path: Path, fields: list[str], row: dict[str, object]) -> None:
    """Durably append one row, writing a header only for a new/empty file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if needs_header:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in fields})
        handle.flush()
        os.fsync(handle.fileno())


def log_processing(
    path: Path,
    operation: str,
    script: str,
    target: str,
    status: str,
    details: str,
) -> None:
    append_csv(path, PROCESSING_LOG_FIELDS, {
        "timestamp": utc_timestamp(),
        "operation": operation,
        "script": script,
        "target": target,
        "status": status,
        "details": details,
    })
