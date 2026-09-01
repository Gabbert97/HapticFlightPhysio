#!/usr/bin/env python3
"""Build physiological and canonical behavioral trial metadata indexes."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from dataset.alignment import align_interval
from dataset.behavioral import read_behavioral_boundary
from dataset.naming import (
    canonical_phase_from_hdf5,
    parse_behavior_filename,
    participant_numeric_id,
)
from dataset.physiology import read_physiology_timing
from dataset.provenance import log_processing

PHYSIO_FIELDS = [
    "participant_id", "participant_numeric_id", "phase", "physio_filename",
    "physio_filepath", "physio_original_filename", "physio_start_timestamp",
    "physio_end_timestamp", "sampling_rate_hz", "n_samples", "physio_duration_s",
    "device_group", "status", "notes",
]

TRIAL_FIELDS = [
    "participant_id", "participant_numeric_id", "phase", "protocol_code",
    "trial_index_file", "trial_number_chronological", "behavior_filename",
    "behavior_filepath", "behavior_start_timestamp", "behavior_end_timestamp",
    "behavior_duration_s", "trial_outcome", "physio_filename", "physio_filepath",
    "physio_original_filename", "physio_start_timestamp", "physio_end_timestamp",
    "sampling_rate_hz", "physio_n_samples", "physio_duration_s", "start_offset_s",
    "end_offset_s", "physio_start_sample", "physio_end_sample", "alignment_status",
    "alignment_valid", "seconds_missing_before", "seconds_missing_after",
    "physiological_overlap_s", "coverage_percent", "include_physio_analysis",
    "physio_exclusion_reason", "notes",
]

PHYSIO_EXCLUSION_REASONS = {
    "partial_start": "partial_physiological_coverage",
    "partial_end": "partial_physiological_coverage",
    "outside_before": "physiology_recording_started_late",
    "outside_after": "physiology_recording_ended_early",
    "missing_physio": "missing_physiology",
    "invalid_timestamp": "invalid_timestamp",
    "empty": "invalid_timestamp",
}


def relative(path: Path) -> str:
    return path.resolve().relative_to(REPOSITORY_ROOT.resolve()).as_posix()


def timestamp(value: datetime | None) -> str:
    return value.isoformat(sep=" ", timespec="milliseconds") if value else ""


def number(value: float | None, places: int = 6) -> str:
    return f"{value:.{places}f}" if value is not None else ""


def discover_physiology(data_dir: Path) -> tuple[list[dict[str, object]], dict[tuple[int, str], dict[str, object]]]:
    rows: list[dict[str, object]] = []
    lookup: dict[tuple[int, str], dict[str, object]] = {}
    for subject_dir in sorted(data_dir.glob("Subject_*"), key=lambda p: participant_numeric_id(p) or 10**9):
        participant_number = participant_numeric_id(subject_dir)
        if participant_number is None:
            continue
        for path in sorted((subject_dir / "raw_bp_signals").glob("*.h5")):
            phase = canonical_phase_from_hdf5(path)
            if phase is None:
                continue
            timing = read_physiology_timing(path)
            key = (participant_number, phase)
            if key in lookup:
                raise ValueError(f"Duplicate physiological mapping for Subject_{participant_number}/{phase}")
            row = {
                "participant_id": f"Subject_{participant_number}",
                "participant_numeric_id": participant_number,
                "phase": phase,
                "physio_filename": f"{phase}.h5",
                "physio_filepath": relative(path),
                "physio_original_filename": path.name,
                "physio_start_timestamp": timestamp(timing.start),
                "physio_end_timestamp": timestamp(timing.end),
                "sampling_rate_hz": f"{timing.sampling_rate_hz:g}",
                "n_samples": timing.n_samples,
                "physio_duration_s": number(timing.duration_s, 3),
                "device_group": timing.device_group,
                "status": timing.status,
                "notes": timing.notes,
                "_timing": timing,
            }
            rows.append(row)
            lookup[key] = row
    return rows, lookup


def discover_behavior(data_dir: Path, physio: dict[tuple[int, str], dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for subject_dir in sorted((data_dir / "Behavioral Data").glob("Subject_*"), key=lambda p: participant_numeric_id(p) or 10**9):
        participant_number = participant_numeric_id(subject_dir)
        if participant_number is None:
            continue
        for path in sorted(subject_dir.glob("*.csv")):
            parsed = parse_behavior_filename(path)
            if parsed is None:
                raise ValueError(f"Unexpected behavioral CSV filename: {relative(path)}")
            protocol_code, phase, trial_index = parsed
            boundary = read_behavioral_boundary(path)
            physio_row = physio.get((participant_number, phase))
            notes = [boundary.notes] if boundary.notes else []

            row: dict[str, object] = {
                "participant_id": f"Subject_{participant_number}",
                "participant_numeric_id": participant_number,
                "phase": phase,
                "protocol_code": protocol_code,
                "trial_index_file": trial_index,
                "trial_number_chronological": "",
                "behavior_filename": path.name,
                "behavior_filepath": relative(path),
                "behavior_start_timestamp": timestamp(boundary.start),
                "behavior_end_timestamp": timestamp(boundary.end),
                "behavior_duration_s": number(boundary.duration_s),
                "trial_outcome": boundary.outcome,
                "physio_filename": "",
                "physio_filepath": "",
                "physio_original_filename": "",
                "physio_start_timestamp": "",
                "physio_end_timestamp": "",
                "sampling_rate_hz": "",
                "physio_n_samples": "",
                "physio_duration_s": "",
                "start_offset_s": "",
                "end_offset_s": "",
                "physio_start_sample": "",
                "physio_end_sample": "",
                "alignment_status": boundary.status,
                "alignment_valid": False,
                "seconds_missing_before": "",
                "seconds_missing_after": "",
                "physiological_overlap_s": "",
                "coverage_percent": "",
                "include_physio_analysis": False,
                "physio_exclusion_reason": "",
                "notes": "",
                "_start": boundary.start,
            }

            if boundary.status == "empty":
                row["alignment_status"] = "empty"
            elif boundary.status == "invalid_timestamp":
                row["alignment_status"] = "invalid_timestamp"
            elif physio_row is None:
                row["alignment_status"] = "missing_physio"
                notes.append("no participant/phase HDF5 match")
            else:
                timing = physio_row["_timing"]
                aligned = align_interval(
                    boundary.start, boundary.end, timing.start, timing.end,
                    timing.sampling_rate_hz,
                )
                row.update({
                    "physio_filename": physio_row["physio_filename"],
                    "physio_filepath": physio_row["physio_filepath"],
                    "physio_original_filename": physio_row["physio_original_filename"],
                    "physio_start_timestamp": physio_row["physio_start_timestamp"],
                    "physio_end_timestamp": physio_row["physio_end_timestamp"],
                    "sampling_rate_hz": physio_row["sampling_rate_hz"],
                    "physio_n_samples": physio_row["n_samples"],
                    "physio_duration_s": physio_row["physio_duration_s"],
                    "start_offset_s": number(aligned.start_offset_s),
                    "end_offset_s": number(aligned.end_offset_s),
                    "physio_start_sample": aligned.start_sample,
                    "physio_end_sample": aligned.end_sample,
                    "alignment_status": aligned.status,
                    "alignment_valid": aligned.valid,
                    "seconds_missing_before": number(aligned.seconds_missing_before),
                    "seconds_missing_after": number(aligned.seconds_missing_after),
                    "physiological_overlap_s": number(aligned.physiological_overlap_s),
                    "coverage_percent": number(aligned.coverage_percent, 3),
                })
                if aligned.notes:
                    notes.append(aligned.notes)
            row["notes"] = "; ".join(notes)
            include_physio = row["alignment_status"] == "valid"
            row["include_physio_analysis"] = include_physio
            row["physio_exclusion_reason"] = (
                "" if include_physio
                else PHYSIO_EXCLUSION_REASONS.get(str(row["alignment_status"]), "invalid_timestamp")
            )
            rows.append(row)

    grouped: dict[tuple[int, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        if row["_start"] is not None:
            grouped[(int(row["participant_numeric_id"]), str(row["phase"]))].append(row)
    for trials in grouped.values():
        trials.sort(key=lambda row: (row["_start"], int(row["trial_index_file"])))
        for chronological, row in enumerate(trials, start=1):
            row["trial_number_chronological"] = chronological
    return rows


def write_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row[field] for field in fields} for row in rows)


def report(rows: list[dict[str, object]]) -> None:
    counts = Counter(str(row["alignment_status"]) for row in rows)
    valid_behavior = sum(row["_start"] is not None for row in rows)
    partial = counts["partial_start"] + counts["partial_end"]
    outside = counts["outside_before"] + counts["outside_after"]
    print("Validation summary")
    print(f"  total behavioral files: {len(rows)}")
    print(f"  valid behavioral files: {valid_behavior}")
    print(f"  empty/header-only files: {counts['empty']}")
    print(f"  aligned trials: {counts['valid']}")
    print(f"  partially aligned trials: {partial}")
    print(f"  completely out-of-bounds trials: {outside}")
    print(f"  missing HDF5 matches: {counts['missing_physio']}")
    print(f"  invalid timestamp files: {counts['invalid_timestamp']}")

    sample_differences = []
    unexpected_mismatches = []
    for row in rows:
        if row["alignment_status"] != "valid":
            continue
        sample_duration = (
            (int(row["physio_end_sample"]) - int(row["physio_start_sample"]))
            / float(row["sampling_rate_hz"])
        )
        difference = abs(sample_duration - float(row["behavior_duration_s"]))
        sample_differences.append(difference)
        if difference > (1.1 / float(row["sampling_rate_hz"])):
            unexpected_mismatches.append(row)
    ordered = sorted(sample_differences)
    mean_difference = sum(ordered) / len(ordered) if ordered else 0.0
    median_difference = (
        ordered[len(ordered) // 2]
        if len(ordered) % 2
        else sum(ordered[len(ordered) // 2 - 1:len(ordered) // 2 + 1]) / 2
    ) if ordered else 0.0
    print("\nSample-bound validation (valid alignments)")
    print(f"  bounds valid: {sum(0 <= int(r['physio_start_sample']) < int(r['physio_end_sample']) <= int(r['physio_n_samples']) for r in rows if r['alignment_status'] == 'valid')}/{counts['valid']}")
    print(f"  mean absolute duration difference: {mean_difference:.9f} s")
    print(f"  median absolute duration difference: {median_difference:.9f} s")
    print(f"  maximum absolute duration difference: {max(ordered, default=0.0):.9f} s")
    print(f"  unexpected mismatches: {len(unexpected_mismatches)}")

    print("\nBy participant and phase")
    grouped: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    for row in rows:
        grouped[(str(row["participant_id"]), str(row["phase"]))][str(row["alignment_status"])] += 1
    for (participant, phase), statuses in grouped.items():
        details = ", ".join(f"{name}={count}" for name, count in sorted(statuses.items()))
        print(f"  {participant}/{phase}: {details}")

    print("\nAlignment exceptions")
    exceptions = [row for row in rows if row["alignment_status"] != "valid"]
    for row in exceptions:
        print(
            f"  {row['participant_id']}/{row['phase']}/index={row['trial_index_file']}: "
            f"{row['alignment_status']} ({row['behavior_filename']}), "
            f"missing_before={row['seconds_missing_before'] or 'n/a'}s, "
            f"missing_after={row['seconds_missing_after'] or 'n/a'}s, "
            f"overlap={row['physiological_overlap_s'] or 'n/a'}s, "
            f"coverage={row['coverage_percent'] or 'n/a'}%"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=REPOSITORY_ROOT / "data")
    parser.add_argument("--metadata-dir", type=Path, default=REPOSITORY_ROOT / "metadata")
    args = parser.parse_args()

    physiology_rows, physiology_lookup = discover_physiology(args.data_dir)
    trial_rows = discover_behavior(args.data_dir, physiology_lookup)
    write_csv(args.metadata_dir / "physiological_recordings.csv", PHYSIO_FIELDS, physiology_rows)
    write_csv(args.metadata_dir / "trial_index.csv", TRIAL_FIELDS, trial_rows)
    processing_log = args.metadata_dir / "processing_log.csv"
    log_processing(
        processing_log, "TRIAL_INDEX_REBUILD", Path(__file__).name,
        "metadata/trial_index.csv", "success",
        f"rebuilt {len(trial_rows)} current-state behavioral trial rows and {len(physiology_rows)} physiological rows",
    )
    status_counts = Counter(str(row["alignment_status"]) for row in trial_rows)
    log_processing(
        processing_log, "ALIGNMENT_VALIDATION", Path(__file__).name,
        "metadata/trial_index.csv", "success",
        "; ".join(f"{key}={value}" for key, value in sorted(status_counts.items())),
    )
    included = sum(bool(row["include_physio_analysis"]) for row in trial_rows)
    log_processing(
        processing_log, "PHYSIO_ANALYSIS_INCLUSION_POLICY", Path(__file__).name,
        "metadata/trial_index.csv", "success",
        f"policy=alignment_status_valid_only; included={included}; excluded={len(trial_rows) - included}",
    )
    print(f"Wrote {relative(args.metadata_dir / 'physiological_recordings.csv')}")
    print(f"Wrote {relative(args.metadata_dir / 'trial_index.csv')}\n")
    report(trial_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
