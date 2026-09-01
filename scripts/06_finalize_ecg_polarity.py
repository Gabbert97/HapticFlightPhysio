#!/usr/bin/env python3
"""Finalize participant ECG polarity decisions in the authoritative QC table."""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from dataset.provenance import log_processing  # noqa: E402

REQUIRED_FIELDS = [
    "participant_id", "ecg_polarity_auto", "ecg_multiplier_auto", "auto_confidence",
    "manual_review_required", "manual_review_status", "ecg_polarity_manual",
    "ecg_multiplier_manual", "ecg_polarity_final", "ecg_multiplier_final",
    "decision_source", "notes",
]


def parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"Invalid boolean value: {value!r}")


def load_table(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fields = list(reader.fieldnames or [])
    if not rows:
        raise ValueError("ECG polarity QC table is empty")
    if "auto_confidence" not in fields:
        fields.append("auto_confidence")
        for row in rows:
            row["auto_confidence"] = row.get("polarity_confidence", "")
    for field in REQUIRED_FIELDS:
        if field not in fields:
            fields.append(field)
        for row in rows:
            row.setdefault(field, "")
    return fields, rows


def write_table_atomic(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def set_manual_decision(row: dict[str, str], decision: str, source: str) -> None:
    multiplier = "1" if decision == "normal" else "-1"
    row["ecg_polarity_manual"] = decision
    row["ecg_multiplier_manual"] = multiplier
    row["ecg_polarity_final"] = decision
    row["ecg_multiplier_final"] = multiplier
    row["manual_review_status"] = "reviewed"
    row["decision_source"] = source
    row["notes"] = "Manual review finalized; correction configured only; raw files unchanged"


def finalize_confident_normals(rows: list[dict[str, str]]) -> list[str]:
    finalized = []
    for row in rows:
        if row["ecg_polarity_auto"] == "normal" and not parse_bool(row["manual_review_required"]):
            row["ecg_polarity_final"] = "normal"
            row["ecg_multiplier_final"] = "1"
            row["manual_review_status"] = "not_required"
            row["decision_source"] = "automatic"
            row["notes"] = "Confident automatic normal; correction configured only; raw files unchanged"
            finalized.append(row["participant_id"])
    return finalized


def prompt_decision(row: dict[str, str], open_plots: bool) -> str | None:
    plot = row.get("qc_review_plot") or (
        f"outputs/qc/ecg_polarity/manual_review/{row['participant_id']}_ecg_polarity_qc.png"
    )
    print(f"\n{row['participant_id']}")
    print(f"Automatic classification: {row['ecg_polarity_auto']}")
    print(f"Confidence: {row['auto_confidence']}")
    print(f"Proposed multiplier: {row['ecg_multiplier_auto'] or 'unresolved'}")
    print(f"QC plot: {plot}")
    if open_plots:
        subprocess.run(["open", str(REPOSITORY_ROOT / plot)], check=False)
    while True:
        value = input("ECG polarity decision [0=normal, 1=inverted, s=skip]: ").strip().lower()
        if value == "0":
            return "normal"
        if value == "1":
            return "inverted"
        if value == "s":
            return None
        print("Invalid value. Enter 0, 1, or s.")


def validate(rows: list[dict[str, str]]) -> tuple[dict[str, int], list[str]]:
    if len(rows) != 21 or len({row["participant_id"] for row in rows}) != 21:
        raise ValueError("Expected exactly 21 unique participants")
    counts = {"normal": 0, "inverted": 0}
    unresolved = []
    for row in rows:
        polarity, multiplier = row["ecg_polarity_final"], row["ecg_multiplier_final"]
        expected = {"normal": "1", "inverted": "-1"}.get(polarity)
        if expected is None or multiplier != expected:
            unresolved.append(row["participant_id"])
        else:
            counts[polarity] += 1
    return counts, unresolved


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", type=int)
    parser.add_argument("--decision", choices=("normal", "inverted"))
    parser.add_argument("--confirm-auto-inverted", action="store_true",
                        help="confirm every metadata-derived automatic inverted review case")
    parser.add_argument("--open-plots", action="store_true")
    parser.add_argument("--metadata", type=Path,
                        default=REPOSITORY_ROOT / "metadata/ecg_polarity_qc.csv")
    args = parser.parse_args()
    if (args.subject is None) != (args.decision is None):
        parser.error("--subject and --decision must be supplied together")

    fields, rows = load_table(args.metadata)
    automatic = finalize_confident_normals(rows)
    confirmed_auto = []
    overrides = []

    if args.confirm_auto_inverted:
        for row in rows:
            if parse_bool(row["manual_review_required"]) and row["ecg_polarity_auto"] == "inverted":
                set_manual_decision(row, "inverted", "manual_confirmed_auto")
                confirmed_auto.append(row["participant_id"])

    if args.subject is not None:
        participant_id = f"Subject_{args.subject}"
        matches = [row for row in rows if row["participant_id"] == participant_id]
        if len(matches) != 1:
            raise ValueError(f"Expected one QC row for {participant_id}, found {len(matches)}")
        row = matches[0]
        if not parse_bool(row["manual_review_required"]):
            raise ValueError(f"Manual review is not required for {participant_id}")
        source = "manual_override" if row["ecg_polarity_auto"] != args.decision else "manual"
        set_manual_decision(row, args.decision, source)
        if source == "manual_override":
            overrides.append(participant_id)

    if args.subject is None and not args.confirm_auto_inverted:
        for row in rows:
            if (parse_bool(row["manual_review_required"])
                    and row["manual_review_status"] != "reviewed"):
                decision = prompt_decision(row, args.open_plots)
                if decision is not None:
                    source = "manual_override" if row["ecg_polarity_auto"] != decision else "manual"
                    set_manual_decision(row, decision, source)
                    if source == "manual_override":
                        overrides.append(row["participant_id"])

    counts, unresolved = validate(rows)
    write_table_atomic(args.metadata, fields, rows)
    log_processing(
        args.metadata.parent / "processing_log.csv", "ECG_POLARITY_FINALIZATION",
        Path(__file__).name, "metadata/ecg_polarity_qc.csv",
        "success" if not unresolved else "partial",
        f"automatic={','.join(automatic)}; manual_confirmed_auto={','.join(confirmed_auto)}; "
        f"manual_overrides={','.join(overrides)}; normal={counts['normal']}; "
        f"inverted={counts['inverted']}; unresolved={len(unresolved)}; "
        "correction_configured=true; raw_files_modified=false",
    )
    print(f"Final counts: normal={counts['normal']}, inverted={counts['inverted']}, "
          f"unresolved={len(unresolved)}")
    if unresolved:
        print(f"Unresolved: {', '.join(unresolved)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
