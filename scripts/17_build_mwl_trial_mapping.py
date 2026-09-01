#!/usr/bin/env python3
"""Map workbook mental-workload observations to final physiological trial keys."""

from __future__ import annotations

import csv
import xml.etree.ElementTree as ET
from pathlib import Path
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parents[1]
WORKBOOK = ROOT / "data/SecondRun_Professional.xlsx"
PHYSIOLOGY = ROOT / "outputs/final_features/physiological_features_all.csv"
OUTPUT = ROOT / "outputs/final_features/mwl_trial_mapping.csv"
NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}

# The workbook's two header rows explicitly place each MW cell after its trial block.
LAYOUT = {
    "pre_test": (("F", "G", "H", "I", "J"), ("K",)),
    "test_1": (
        (("L", "M", "N", "O"), ("Q", "R", "S", "T"), ("V", "W", "X", "Y")),
        ("P", "U", "Z"),
    ),
    "test_2": (
        (("AA", "AB", "AC", "AD"), ("AF", "AG", "AH", "AI"), ("AK", "AL", "AM", "AN")),
        ("AE", "AJ", "AO"),
    ),
    "test_3": (
        (("AP", "AQ", "AR", "AS"), ("AU", "AV", "AW", "AX"), ("AZ", "BA", "BB", "BC")),
        ("AT", "AY", "BD"),
    ),
    "evaluation": (("BE", "BF", "BG", "BH", "BI"), ("BJ",)),
}
FIELDS = (
    "participant_id", "group", "phase", "mwl_measure_index", "mwl_value",
    "first_trial", "last_trial", "number_of_trials", "trial_numbers",
    "mapping_status",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def workbook_rows() -> list[dict[str, str]]:
    with ZipFile(WORKBOOK) as archive:
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        sheets = workbook.findall(".//m:sheets/m:sheet", NS)
        if len(sheets) != 1 or sheets[0].get("name") != "Run 2 Data":
            raise ValueError("Expected the single workbook sheet 'Run 2 Data'")
        shared_root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
        shared = [
            "".join(node.text or "" for node in item.iterfind(".//m:t", NS))
            for item in shared_root.findall("m:si", NS)
        ]
        sheet = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))

    def values(row: ET.Element) -> dict[str, str]:
        result = {}
        for cell in row.findall("m:c", NS):
            reference = cell.get("r", "")
            column = "".join(char for char in reference if char.isalpha())
            node = cell.find("m:v", NS)
            value = "" if node is None else (node.text or "")
            if cell.get("t") == "s" and value:
                value = shared[int(value)]
            result[column] = value.strip()
        return result

    xml_rows = sheet.findall(".//m:sheetData/m:row", NS)
    section, header = values(xml_rows[0]), values(xml_rows[1])
    expected_sections = {
        "F": "PreTest", "L": "Test 1 (Easy)", "AA": "Test 2 (Medium)",
        "AP": "Test 3 (Hard)", "BE": "Evaluation",
    }
    for column, label in expected_sections.items():
        if section.get(column) != label:
            raise ValueError(f"Unexpected workbook section at {column}: {section.get(column)!r}")
    for phase, (blocks, mw_columns) in LAYOUT.items():
        block_list = (blocks,) if phase in {"pre_test", "evaluation"} else blocks
        for block in block_list:
            for index, column in enumerate(block, 1):
                if header.get(column) != f"Trial {index if len(block_list)==1 else block_list.index(block)*4+index}":
                    raise ValueError(f"Unexpected trial header at {column}: {header.get(column)!r}")
        for column in mw_columns:
            if header.get(column) != "MW":
                raise ValueError(f"Expected MW header at {column}")
    return [values(row) for row in xml_rows[2:]]


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {OUTPUT}")
    final_rows = read_csv(PHYSIOLOGY)
    final_trials: dict[tuple[str, str], list[int]] = {}
    for row in final_rows:
        if row["phase"] == "baseline":
            continue
        final_trials.setdefault((row["participant_id"], row["phase"]), []).append(
            int(float(row["trial_number"]))
        )
    for trials in final_trials.values():
        trials.sort()

    output = []
    missing_ratings = []
    seen = set()
    for source in workbook_rows():
        digits = "".join(char for char in source.get("A", "") if char.isdigit())
        if not digits or int(digits) > 21:
            continue
        participant_id = f"Subject_{int(digits)}"
        if participant_id in seen:
            raise ValueError(f"Duplicate workbook participant: {participant_id}")
        seen.add(participant_id)
        group = {"NOHA": "NoHA", "HA": "Haptic"}.get(source.get("B", "").upper())
        if group is None:
            raise ValueError(f"Unknown group for {participant_id}: {source.get('B')!r}")

        for phase, (blocks, mw_columns) in LAYOUT.items():
            phase_trials = final_trials[(participant_id, phase)]
            if phase in {"pre_test", "evaluation"}:
                rating = source.get(mw_columns[0], "")
                if not rating:
                    missing_ratings.append((participant_id, phase, 1, phase_trials))
                    continue
                mapped = phase_trials
                status = "complete_phase"
                block_list = (blocks,)
            else:
                block_list = blocks
                for measure_index, (block, mw_column) in enumerate(zip(block_list, mw_columns), 1):
                    rating = source.get(mw_column, "")
                    source_trial_count = sum(bool(source.get(column, "")) for column in block)
                    nominal = range((measure_index - 1) * 4 + 1, measure_index * 4 + 1)
                    mapped = [trial for trial in phase_trials if trial in nominal]
                    if not rating:
                        if source_trial_count:
                            missing_ratings.append((participant_id, phase, measure_index, mapped))
                        continue
                    if len(mapped) == 4:
                        status = "complete_4_trial_block"
                    elif source_trial_count < 4 and len(mapped) == source_trial_count:
                        status = "incomplete_source_block"
                    elif len(mapped) < source_trial_count:
                        status = "incomplete_physiological_coverage"
                    else:
                        status = "incomplete_4_trial_block"
                    output.append({
                        "participant_id": participant_id, "group": group,
                        "phase": phase, "mwl_measure_index": measure_index,
                        "mwl_value": float(rating),
                        "first_trial": mapped[0] if mapped else "",
                        "last_trial": mapped[-1] if mapped else "",
                        "number_of_trials": len(mapped),
                        "trial_numbers": ";".join(map(str, mapped)),
                        "mapping_status": status,
                    })
                continue

            output.append({
                "participant_id": participant_id, "group": group, "phase": phase,
                "mwl_measure_index": 1, "mwl_value": float(rating),
                "first_trial": mapped[0] if mapped else "",
                "last_trial": mapped[-1] if mapped else "",
                "number_of_trials": len(mapped),
                "trial_numbers": ";".join(map(str, mapped)),
                "mapping_status": status,
            })

    if seen != {f"Subject_{number}" for number in range(1, 22)}:
        raise ValueError("Workbook participant mapping is incomplete")
    with OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(output)

    covered = {
        (row["participant_id"], row["phase"], trial)
        for row in output
        for trial in map(int, str(row["trial_numbers"]).split(";")) if str(row["trial_numbers"])
    }
    all_trials = {
        (participant, phase, trial)
        for (participant, phase), trials in final_trials.items() for trial in trials
    }
    print("mwl_observations", len(output))
    print("by_phase", {phase: sum(row["phase"] == phase for row in output) for phase in LAYOUT})
    print("mapping_status", {
        status: sum(row["mapping_status"] == status for row in output)
        for status in sorted({str(row["mapping_status"]) for row in output})
    })
    print("missing_ratings", missing_ratings)
    print("unmapped_final_trials", sorted(all_trials - covered))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
