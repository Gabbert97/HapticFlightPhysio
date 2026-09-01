"""Classify behavioral intervals relative to physiological recordings."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class Alignment:
    start_offset_s: float
    end_offset_s: float
    start_sample: int
    end_sample: int
    status: str
    valid: bool
    seconds_missing_before: float
    seconds_missing_after: float
    physiological_overlap_s: float
    coverage_percent: float
    notes: str = ""


def align_interval(
    behavior_start: datetime,
    behavior_end: datetime,
    physio_start: datetime,
    physio_end: datetime,
    sampling_rate_hz: float,
) -> Alignment:
    start_offset = (behavior_start - physio_start).total_seconds()
    end_offset = (behavior_end - physio_start).total_seconds()
    start_sample = round(start_offset * sampling_rate_hz)
    end_sample = round(end_offset * sampling_rate_hz)

    behavior_duration = (behavior_end - behavior_start).total_seconds()
    overlap_start = max(behavior_start, physio_start)
    overlap_end = min(behavior_end, physio_end)
    overlap_s = max(0.0, (overlap_end - overlap_start).total_seconds())
    missing_before = max(0.0, (min(behavior_end, physio_start) - behavior_start).total_seconds())
    missing_after = max(0.0, (behavior_end - max(behavior_start, physio_end)).total_seconds())
    coverage = 100.0 * overlap_s / behavior_duration if behavior_duration > 0 else 0.0

    start_inside = physio_start <= behavior_start <= physio_end
    end_inside = physio_start <= behavior_end <= physio_end

    notes = ""
    if start_inside and end_inside:
        status = "valid"
    elif behavior_start < physio_start and end_inside:
        status = "partial_start"
    elif start_inside and behavior_end > physio_end:
        status = "partial_end"
    elif behavior_end < physio_start:
        status = "outside_before"
    elif behavior_start > physio_end:
        status = "outside_after"
    else:
        # The requested vocabulary has no dedicated label for an interval that
        # spans both recording boundaries. Keep it non-valid and explicit.
        status = "invalid_timestamp"
        notes = "behavioral interval spans both physiological boundaries"

    return Alignment(
        start_offset,
        end_offset,
        start_sample,
        end_sample,
        status,
        status == "valid",
        missing_before,
        missing_after,
        overlap_s,
        coverage,
        notes,
    )
