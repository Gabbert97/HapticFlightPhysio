#!/usr/bin/env python3
"""Dry-run or apply collision-safe HDF5 filename normalization."""

from __future__ import annotations

import argparse
import csv
import os
import sys
import uuid
from collections import Counter
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from dataset.naming import CANONICAL_PHASES, canonical_phase_from_hdf5, participant_numeric_id

FIELDS = [
    "participant_id", "original_filepath", "original_filename", "canonical_phase",
    "new_filename", "new_filepath", "rename_needed", "collision", "status",
]


def relative(path: Path) -> str:
    return path.resolve().relative_to(REPOSITORY_ROOT.resolve()).as_posix()


def same_existing_file(left: Path, right: Path) -> bool:
    try:
        return left.exists() and right.exists() and os.path.samefile(left, right)
    except OSError:
        return False


def discover(data_dir: Path) -> tuple[list[dict[str, object]], list[str]]:
    rows: list[dict[str, object]] = []
    errors: list[str] = []
    for subject_dir in sorted(data_dir.glob("Subject_*"), key=lambda p: participant_numeric_id(p) or 10**9):
        participant_number = participant_numeric_id(subject_dir)
        if participant_number is None:
            continue
        files = sorted((subject_dir / "raw_bp_signals").glob("*.h5"))
        mapped = [canonical_phase_from_hdf5(path) for path in files]
        counts = Counter(phase for phase in mapped if phase)
        missing = sorted(set(CANONICAL_PHASES) - set(counts))
        duplicates = sorted(phase for phase, count in counts.items() if count != 1)
        if missing:
            errors.append(f"Subject_{participant_number}: missing {', '.join(missing)}")
        if duplicates:
            errors.append(f"Subject_{participant_number}: duplicate mappings {', '.join(duplicates)}")

        for path, phase in zip(files, mapped):
            target = path.with_name(f"{phase}.h5") if phase else path
            rename_needed = phase is not None and path.name != target.name
            collision = bool(
                rename_needed and target.exists() and not same_existing_file(path, target)
            )
            if phase is None:
                status = "unexpected_hdf5"
                errors.append(f"Unexpected HDF5 file: {relative(path)}")
            elif collision:
                status = "collision"
                errors.append(f"Target collision: {relative(path)} -> {relative(target)}")
            elif rename_needed:
                status = "would_rename"
            else:
                status = "already_canonical"
            rows.append({
                "participant_id": f"Subject_{participant_number}",
                "original_filepath": relative(path),
                "original_filename": path.name,
                "canonical_phase": phase or "",
                "new_filename": target.name if phase else "",
                "new_filepath": relative(target) if phase else "",
                "rename_needed": rename_needed,
                "collision": collision,
                "status": status,
                "_source": path,
                "_target": target,
            })
    return rows, errors


def write_log(rows: list[dict[str, object]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows({field: row[field] for field in FIELDS} for row in rows)


def apply_renames(rows: list[dict[str, object]]) -> int:
    pending = [row for row in rows if row["rename_needed"]]
    # Two-stage moves safely support case-only changes on case-insensitive filesystems.
    staged: list[tuple[Path, Path, Path]] = []
    try:
        for row in pending:
            source, target = row["_source"], row["_target"]
            temporary = source.with_name(f".{source.name}.rename-{uuid.uuid4().hex}.tmp")
            source.rename(temporary)
            staged.append((source, temporary, target))
        for source, temporary, target in staged:
            if target.exists():
                raise FileExistsError(f"Target appeared during rename: {target}")
            temporary.rename(target)
    except Exception:
        # Restore every still-staged temporary file where safely possible.
        for source, temporary, target in reversed(staged):
            if temporary.exists() and not source.exists():
                temporary.rename(source)
            elif target.exists() and not source.exists():
                target.rename(source)
        raise
    return len(pending)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="apply validated renames")
    parser.add_argument("--data-dir", type=Path, default=REPOSITORY_ROOT / "data")
    parser.add_argument(
        "--log", type=Path, default=REPOSITORY_ROOT / "metadata" / "hdf5_rename_log.csv"
    )
    args = parser.parse_args()

    rows, errors = discover(args.data_dir)
    public_rows = [{field: row[field] for field in FIELDS} for row in rows]
    writer = csv.DictWriter(sys.stdout, fieldnames=FIELDS)
    writer.writeheader()
    writer.writerows(public_rows)

    pending = sum(bool(row["rename_needed"]) for row in rows)
    print(f"\nHDF5 files: {len(rows)}; pending renames: {pending}; errors: {len(errors)}")
    if errors:
        write_log(rows, args.log)
        print(f"Audit log: {relative(args.log)}")
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        print("Aborted: preflight validation failed.", file=sys.stderr)
        return 2
    if not args.apply:
        # Preserve the original-name audit after a successful apply. A later
        # idempotence check must not erase the provenance it is meant to test.
        if pending or not args.log.exists():
            write_log(rows, args.log)
            print(f"Audit log written: {relative(args.log)}")
        else:
            print(f"Existing audit log retained: {relative(args.log)}")
        print("DRY RUN: no HDF5 files renamed. Re-run with --apply to rename.")
        return 0

    renamed = apply_renames(rows)
    for row in rows:
        if row["rename_needed"]:
            row["status"] = "renamed"
    write_log(rows, args.log)
    print(f"Audit log written: {relative(args.log)}")
    print(f"Applied {renamed} HDF5 rename(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
