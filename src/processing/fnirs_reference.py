"""Participant-specific fNIRS reference construction."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from multimodalphysiokit.io import read_biosignalsplux_hdf5
from multimodalphysiokit.processors import FNIRSProcessor
from multimodalphysiokit.processors.fnirs import _differential_pathlength_factor

from dataset.baseline_events import read_baseline_rising_edges
from processing.common import segment_signal


def build_fnirs_reference(participant_id: str, age_years: float | None,
                          baseline_hdf5: Path, source_hdf5: str) -> dict[str, object]:
    events, sampling_rate, n_samples = read_baseline_rising_edges(baseline_hdf5)
    row: dict[str, object] = {
        "participant_id": participant_id, "age_years": "" if age_years is None else age_years,
        "age_source": "data/SecondRun_Professional.xlsx:Age", "baseline_hdf5": source_hdf5,
        "first_event_sample": "", "first_event_time_s": "", "reference_start_sample": "",
        "reference_end_sample": "", "reference_start_time_s": "", "reference_end_time_s": "",
        "reference_duration_s": "", "sampling_rate": sampling_rate, "red_reference": "",
        "infrared_reference": "", "reference_method":
        "FNIRSProcessor.compute_baseline_values: filtered-current mean after trimming 10s per end",
        "dpf_660nm": "", "dpf_860nm": "", "fnirs_reference_status": "pending",
    }
    if not events:
        row["fnirs_reference_status"] = "invalid_missing_first_rising_edge"
        return row
    first = events[0]
    start = first.event_sample + round(30 * sampling_rate)
    target_end = first.event_sample + round(210 * sampling_rate)
    end = min(target_end, n_samples)
    row.update(first_event_sample=first.event_sample, first_event_time_s=first.event_time_s,
               reference_start_sample=start, reference_end_sample=end,
               reference_start_time_s=start / sampling_rate,
               reference_end_time_s=end / sampling_rate,
               reference_duration_s=(end - start) / sampling_rate)
    duration = (end - start) / sampling_rate
    row["reference_duration_s"] = duration
    if start < 0 or end > n_samples or end <= start or duration < 60:
        row["fnirs_reference_status"] = "invalid_insufficient_baseline_window"
        return row
    if age_years is None or not np.isfinite(age_years) or age_years <= 0:
        row["fnirs_reference_status"] = "invalid_missing_age"
        return row
    recording = read_biosignalsplux_hdf5(baseline_hdf5)
    red = segment_signal(recording.get_signal("fnirs_red"), start, end)
    infrared = segment_signal(recording.get_signal("fnirs_infrared"), start, end)
    processor = FNIRSProcessor(age_years=age_years)
    red_reference, infrared_reference = processor.compute_baseline_values(red, infrared)
    if not (np.isfinite(red_reference) and red_reference > 0 and
            np.isfinite(infrared_reference) and infrared_reference > 0):
        row["fnirs_reference_status"] = "invalid_reference_currents"
        return row
    row.update(red_reference=red_reference, infrared_reference=infrared_reference,
               dpf_660nm=_differential_pathlength_factor(660, age_years),
               dpf_860nm=_differential_pathlength_factor(860, age_years),
               fnirs_reference_status=("valid" if np.isclose(duration, 180) else "valid_shortened"))
    return row
