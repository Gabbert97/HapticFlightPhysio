#!/usr/bin/env python3
"""Verify, log, and optionally remove conclusively unusable behavioral CSV files."""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from dataset.behavioral import TIMESTAMP_FORMAT
from dataset.naming import parse_behavior_filename, participant_numeric_id
from dataset.provenance import append_csv, log_processing, utc_timestamp

REMOVAL_FIELDS = [
    "participant_id", "phase", "protocol_code", "original_filename",
    "original_filepath", "file_size_bytes", "classification", "number_of_rows",
    "number_of_valid_timestamps", "removal_reason", "removal_timestamp", "script",
    "operation_status",
]

CANDIDATE_FIELDS = [
    "participant_id", "phase", "protocol_code", "filename", "filepath",
    "file_size_bytes", "number_of_rows", "number_of_valid_timestamps", "classification",
]


@dataclass(frozen=True)
class ScanResult:
    participant_id: str
    phase: str
    protocol_code: int
    filename: str
    filepath: str
    file_size_bytes: int
    number_of_rows: int
    number_of_valid_timestamps: int
    classification: str
    path: Path


def relative(path: Path) -> str:
    return path.resolve().relative_to(REPOSITORY_ROOT.resolve()).as_posix()


def scan_file(path: Path, participant_number: int) -> ScanResult:
    parsed = parse_behavior_filename(path)
    if parsed is None:
        raise ValueError(f"Unexpected behavioral filename: {relative(path)}")
    protocol_code, phase, _ = parsed
    size = path.stat().st_size
    if size == 0:
        return ScanResult(
            f"Subject_{participant_number}", phase, protocol_code, path.name,
            relative(path), size, 0, 0, "zero_byte", path,
        )

    row_count = 0
    valid_timestamps = 0
    non_footer_rows_without_timestamp = 0
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration:
            raise ValueError(f"Non-zero file has no CSV header: {relative(path)}")
        if "TimeStamp" not in header:
            raise ValueError(f"CSV header lacks TimeStamp: {relative(path)}")
        timestamp_index = header.index("TimeStamp")
        for row in reader:
            row_count += 1
            value = row[timestamp_index].strip() if len(row) > timestamp_index else ""
            if value:
                try:
                    datetime.strptime(value, TIMESTAMP_FORMAT)
                    valid_timestamps += 1
                except ValueError:
                    non_footer_rows_without_timestamp += 1
            else:
                nonempty = [cell.strip() for cell in row if cell.strip()]
                if nonempty and not nonempty[0].lower().startswith("trial "):
                    non_footer_rows_without_timestamp += 1

    if valid_timestamps:
        classification = "valid"
    elif row_count == 0:
        classification = "header_only"
    elif non_footer_rows_without_timestamp == 0:
        # A header plus only a recognized trial-outcome footer has no usable data.
        classification = "header_only"
    else:
        raise ValueError(
            f"Safety abort: zero valid timestamps but unrecognized data rows in {relative(path)}"
        )
    return ScanResult(
        f"Subject_{participant_number}", phase, protocol_code, path.name,
        relative(path), size, row_count, valid_timestamps, classification, path,
    )


def scan_all(behavior_root: Path) -> list[ScanResult]:
    results: list[ScanResult] = []
    for subject_dir in sorted(behavior_root.glob("Subject_*"), key=lambda p: participant_numeric_id(p) or 10**9):
        participant_number = participant_numeric_id(subject_dir)
        if participant_number is None:
            continue
        for path in sorted(subject_dir.glob("*.csv")):
            results.append(scan_file(path, participant_number))
    return results


def print_candidates(candidates: list[ScanResult]) -> None:
    writer = csv.DictWriter(sys.stdout, fieldnames=CANDIDATE_FIELDS)
    writer.writeheader()
    for result in candidates:
        writer.writerow({field: getattr(result, field) for field in CANDIDATE_FIELDS})


def prior_removals(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {row["original_filepath"] for row in csv.DictReader(handle)}


def remove_verified(
    candidates: list[ScanResult], behavior_root: Path, removal_log: Path,
    processing_log: Path,
) -> list[ScanResult]:
    root = behavior_root.resolve()
    already_logged = prior_removals(removal_log)
    removed: list[ScanResult] = []
    for original in candidates:
        path = original.path
        resolved = path.resolve()
        if not resolved.is_relative_to(root):
            raise ValueError(f"Refusing path outside behavioral root: {path}")
        if path.suffix.lower() != ".csv" or path.suffix.lower() == ".h5":
            raise ValueError(f"Refusing non-behavioral CSV target: {path}")
        if not path.exists():
            raise FileNotFoundError(f"Candidate disappeared before deletion: {path}")

        # Mandatory immediate revalidation prevents deletion after any intervening change.
        participant_number = participant_numeric_id(path.parent)
        current = scan_file(path, participant_number)
        if current.classification not in {"zero_byte", "header_only"}:
            print(f"SKIPPED (now contains valid timestamps): {current.filepath}")
            continue
        if current.number_of_valid_timestamps != 0:
            print(f"SKIPPED (valid timestamps found): {current.filepath}")
            continue

        removal_time = utc_timestamp()
        if current.filepath not in already_logged:
            append_csv(removal_log, REMOVAL_FIELDS, {
                "participant_id": current.participant_id,
                "phase": current.phase,
                "protocol_code": current.protocol_code,
                "original_filename": current.filename,
                "original_filepath": current.filepath,
                "file_size_bytes": current.file_size_bytes,
                "classification": current.classification,
                "number_of_rows": current.number_of_rows,
                "number_of_valid_timestamps": current.number_of_valid_timestamps,
                "removal_reason": "conclusively unusable behavioral CSV with zero valid TimeStamp values",
                "removal_timestamp": removal_time,
                "script": Path(__file__).name,
                "operation_status": "removed",
            })
            already_logged.add(current.filepath)
        path.unlink()
        removed.append(current)
        print(f"DELETED: {current.filepath}")
        log_processing(
            processing_log, "BEHAVIORAL_EMPTY_FILE_REMOVAL", Path(__file__).name,
            current.filepath, "success", current.classification,
        )
    return removed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="delete verified candidates")
    parser.add_argument(
        "--behavior-dir", type=Path,
        default=REPOSITORY_ROOT / "data" / "Behavioral Data",
    )
    parser.add_argument("--metadata-dir", type=Path, default=REPOSITORY_ROOT / "metadata")
    args = parser.parse_args()
    processing_log = args.metadata_dir / "processing_log.csv"
    removal_log = args.metadata_dir / "removed_behavioral_files.csv"

    results = scan_all(args.behavior_dir)
    candidates = [r for r in results if r.classification in {"zero_byte", "header_only"}]
    print("Deletion candidates (fresh raw-file scan)")
    print_candidates(candidates)
    print(f"\nScanned {len(results)} files; candidates: {len(candidates)}")
    log_processing(
        processing_log, "BEHAVIORAL_CLEANUP_SCAN", Path(__file__).name,
        relative(args.behavior_dir), "success",
        f"files={len(results)}; candidates={len(candidates)}; apply={args.apply}",
    )

    if not args.apply:
        print("DRY RUN: no behavioral files deleted. Re-run with --apply to remove candidates.")
        return 0

    removed = remove_verified(candidates, args.behavior_dir, removal_log, processing_log)
    remaining = scan_all(args.behavior_dir)
    remaining_candidates = [r for r in remaining if r.classification in {"zero_byte", "header_only"}]
    zero_remaining = sum(r.classification == "zero_byte" for r in remaining_candidates)
    header_remaining = sum(r.classification == "header_only" for r in remaining_candidates)
    print(f"\nRemoved: {len(removed)}; behavioral files remaining: {len(remaining)}")
    print(f"Zero-byte behavioral files remaining: {zero_remaining}")
    print(f"Header-only behavioral files remaining: {header_remaining}")
    if remaining_candidates:
        print("Remaining unusable files (not automatically deleted):")
        print_candidates(remaining_candidates)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
