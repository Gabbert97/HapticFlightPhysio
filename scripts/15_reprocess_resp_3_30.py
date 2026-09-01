#!/usr/bin/env python3
"""Reprocess baseline and experimental RESP with a 3--30 breaths/min range.

All outputs are isolated from the previously validated feature tables.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
from multimodalphysiokit.io import read_biosignalsplux_hdf5
from multimodalphysiokit.processors import RespirationProcessor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from processing.common import participant_metadata_from_xlsx, read_csv_rows  # noqa: E402


MIN_RATE = 3.0
MAX_RATE = 30.0
OUTPUT = ROOT / "outputs/features/resp/rate_3_30"
FEATURE_ID = (
    "participant_id", "group", "phase", "trial_number", "source_hdf5",
    "start_sample", "end_sample", "duration_s", "processing_status",
    "processing_reason", "min_breathing_rate", "max_breathing_rate",
)


def truth(value: str) -> bool:
    return value.strip().lower() == "true"


def write_rows(path: Path, rows: list[dict[str, object]], fields=FEATURE_ID) -> None:
    columns = list(fields) + sorted({key for row in rows for key in row} - set(fields))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def process_interval(
    participant_id: str,
    group: str,
    phase: str,
    trial_number: str,
    source_hdf5: str,
    start_sample: int,
    end_sample: int,
) -> dict[str, object]:
    recording = read_biosignalsplux_hdf5(ROOT / source_hdf5)
    sampling_rate = recording.get_signal("rip").sampling_frequency
    start_seconds = start_sample / sampling_rate
    end_seconds = np.nextafter(end_sample / sampling_rate, -np.inf)
    recording.set_phases(np.asarray([[start_seconds], [end_seconds]]), ["selected"])
    row: dict[str, object] = {
        "participant_id": participant_id,
        "group": group,
        "phase": phase,
        "trial_number": trial_number,
        "source_hdf5": source_hdf5,
        "start_sample": start_sample,
        "end_sample": end_sample,
        "duration_s": (end_sample - start_sample) / sampling_rate,
        "processing_status": "pending",
        "processing_reason": "",
        "min_breathing_rate": MIN_RATE,
        "max_breathing_rate": MAX_RATE,
    }
    try:
        processor = RespirationProcessor(
            min_breathing_rate=MIN_RATE,
            max_breathing_rate=MAX_RATE,
        )
        result = recording.process_phases(
            processor,
            ("rip",),
            phases=["selected"],
            process_kwargs={"show_plots": False, "save_plots": False},
        )["selected"]
        row.update(processing_status="success")
        row.update({name: float(value) for name, value in result.features.items()})
    except Exception as exc:  # preserve independent observation-level failures
        row.update(
            processing_status="failed",
            processing_reason=f"{type(exc).__name__}: {exc}",
        )
    return row


def comparison_rows(new_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    old = read_csv_rows(ROOT / "metadata/features_resp.csv")
    key = lambda row: (
        row["participant_id"], row["phase"], str(int(row["trial_number"]))
    )
    new_by_key = {
        key(row): row for row in new_rows if row["processing_status"] == "success"
    }
    rows = []
    metadata = set(FEATURE_ID)
    for previous in old:
        current = new_by_key[key(previous)]
        for feature in sorted(set(previous) & set(current) - metadata):
            try:
                old_value = float(previous[feature])
                new_value = float(current[feature])
            except (TypeError, ValueError):
                continue
            difference = abs(old_value - new_value)
            rows.append(
                {
                    "feature": feature,
                    "participant_id": previous["participant_id"],
                    "phase": previous["phase"],
                    "trial_number": previous["trial_number"],
                    "old_value": old_value,
                    "new_value": new_value,
                    "absolute_difference": difference,
                    "equal_or_close": bool(
                        np.isclose(old_value, new_value, rtol=1e-9, atol=1e-11)
                    ),
                }
            )
    return rows


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {OUTPUT}")
    OUTPUT.mkdir(parents=True)

    people = participant_metadata_from_xlsx(ROOT / "data/SecondRun_Professional.xlsx")
    groups = {row["participant_id"]: row["group"] for row in people}
    physiology = {
        (row["participant_id"], row["phase"]): row
        for row in read_csv_rows(ROOT / "metadata/physiological_recordings.csv")
    }
    intervals = {
        row["participant_id"]: row
        for row in read_csv_rows(ROOT / "outputs/features/baseline_interval_manifest.csv")
    }

    def baseline_row(participant_id: str) -> dict[str, object]:
        source = physiology[(participant_id, "baseline")]
        sampling_rate = float(source["sampling_rate_hz"])
        interval = intervals[participant_id]
        return process_interval(
            participant_id,
            groups[participant_id],
            "baseline",
            "",
            source["physio_filepath"],
            round(float(interval["baseline_start_s"]) * sampling_rate),
            round(float(interval["baseline_end_s"]) * sampling_rate),
        )

    # Required first validation; reuse this result rather than processing it twice.
    subject6 = baseline_row("Subject_6")
    if subject6["processing_status"] != "success":
        raise RuntimeError(f"Subject_6 preflight failed: {subject6['processing_reason']}")

    baseline = [subject6]
    baseline.extend(
        baseline_row(person["participant_id"])
        for person in people
        if person["participant_id"] != "Subject_6"
    )

    trials = [
        row for row in read_csv_rows(ROOT / "metadata/trial_index.csv")
        if truth(row["include_physio_analysis"])
    ]
    experimental = []
    trial_groups: dict[tuple[str, str], list[dict[str, str]]] = {}
    for trial in trials:
        trial_groups.setdefault((trial["participant_id"], trial["phase"]), []).append(trial)
    for (participant_id, phase), selected in trial_groups.items():
        selected.sort(key=lambda row: int(row["trial_number_chronological"]))
        source_hdf5 = selected[0]["physio_filepath"]
        recording = read_biosignalsplux_hdf5(ROOT / source_hdf5)
        sampling_rate = recording.get_signal("rip").sampling_frequency
        labels = [
            f"trial_{int(row['trial_number_chronological']):02d}" for row in selected
        ]
        recording.set_phases(
            np.asarray([
                [int(row["physio_start_sample"]) / sampling_rate for row in selected],
                [
                    np.nextafter(int(row["physio_end_sample"]) / sampling_rate, -np.inf)
                    for row in selected
                ],
            ]),
            labels,
        )
        processor = RespirationProcessor(
            min_breathing_rate=MIN_RATE,
            max_breathing_rate=MAX_RATE,
        )
        for trial, label in zip(selected, labels):
            start_sample = int(trial["physio_start_sample"])
            end_sample = int(trial["physio_end_sample"])
            row: dict[str, object] = {
                "participant_id": participant_id,
                "group": groups[participant_id],
                "phase": phase,
                "trial_number": trial["trial_number_chronological"],
                "source_hdf5": source_hdf5,
                "start_sample": start_sample,
                "end_sample": end_sample,
                "duration_s": (end_sample - start_sample) / sampling_rate,
                "processing_status": "pending",
                "processing_reason": "",
                "min_breathing_rate": MIN_RATE,
                "max_breathing_rate": MAX_RATE,
            }
            try:
                result = recording.process_phases(
                    processor,
                    ("rip",),
                    phases=[label],
                    process_kwargs={"show_plots": False, "save_plots": False},
                )[label]
                row.update(processing_status="success")
                row.update({name: float(value) for name, value in result.features.items()})
            except Exception as exc:
                row.update(
                    processing_status="failed",
                    processing_reason=f"{type(exc).__name__}: {exc}",
                )
            experimental.append(row)

    baseline_success = [row for row in baseline if row["processing_status"] == "success"]
    experimental_success = [
        row for row in experimental if row["processing_status"] == "success"
    ]
    write_rows(OUTPUT / "baseline_features.csv", baseline)
    write_rows(OUTPUT / "experimental_features.csv", experimental_success)
    write_rows(OUTPUT / "features_all.csv", baseline + experimental_success)
    write_rows(OUTPUT / "processing_manifest.csv", baseline + experimental)

    comparisons = comparison_rows(experimental)
    comparison_fields = (
        "feature", "participant_id", "phase", "trial_number", "old_value",
        "new_value", "absolute_difference", "equal_or_close",
    )
    write_rows(OUTPUT / "existing_success_comparison.csv", comparisons, comparison_fields)

    recovered = [
        row for row in experimental
        if row["processing_status"] == "success"
        and not any(
            old["participant_id"] == row["participant_id"]
            and old["phase"] == row["phase"]
            and int(old["trial_number"]) == int(row["trial_number"])
            for old in read_csv_rows(ROOT / "metadata/features_resp.csv")
        )
    ]
    summary = [{
        "min_breathing_rate": MIN_RATE,
        "max_breathing_rate": MAX_RATE,
        "baseline_attempted": len(baseline),
        "baseline_successful": len(baseline_success),
        "baseline_failed": len(baseline) - len(baseline_success),
        "experimental_attempted": len(experimental),
        "experimental_successful": len(experimental_success),
        "experimental_failed": len(experimental) - len(experimental_success),
        "experimental_recovered": len(recovered),
        "existing_values_compared": len(comparisons),
        "existing_values_equal_or_close": sum(row["equal_or_close"] for row in comparisons),
        "maximum_existing_absolute_difference": max(
            (row["absolute_difference"] for row in comparisons), default=0.0
        ),
    }]
    write_rows(OUTPUT / "validation_summary.csv", summary, tuple(summary[0]))
    print("Subject_6", subject6["processing_status"], {
        name: subject6[name] for name in subject6 if name.startswith("respiration_")
    })
    print("baseline", len(baseline_success), "success", len(baseline)-len(baseline_success), "failed")
    print("experimental", len(experimental_success), "success", len(experimental)-len(experimental_success), "failed")
    print("recovered", len(recovered))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
