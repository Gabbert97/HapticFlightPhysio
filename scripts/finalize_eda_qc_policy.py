#!/usr/bin/env python3
"""Apply the reviewed EDA inclusion policy without changing automatic QC."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
from dataset.provenance import log_processing  # noqa: E402


def main() -> int:
    path = REPOSITORY_ROOT / "metadata/eda_saturation_qc.csv"
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows, fields = list(reader), list(reader.fieldnames or [])
    if "eda_processing_decision" not in fields:
        fields.append("eda_processing_decision")
    for row in rows:
        status = row["eda_saturation_status"]
        if status == "saturated":
            row.update(eda_include_for_processing="False", manual_review_required="False",
                       manual_review_status="not_required",
                       eda_processing_decision="exclude_saturated")
        elif status == "possible_saturation":
            row.update(eda_include_for_processing="True", manual_review_required="False",
                       manual_review_status="accepted_for_processing",
                       eda_processing_decision="include_with_qc_warning")
        elif status == "ok":
            row.update(eda_include_for_processing="True", manual_review_required="False",
                       manual_review_status="not_required", eda_processing_decision="include")
        else:
            raise ValueError(f"Unknown EDA saturation status: {status}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    included = sum(row["eda_include_for_processing"] == "True" for row in rows)
    possible = sum(row["eda_saturation_status"] == "possible_saturation" for row in rows)
    saturated = sum(row["eda_saturation_status"] == "saturated" for row in rows)
    log_processing(
        REPOSITORY_ROOT / "metadata/processing_log.csv",
        "EDA_QC_PROCESSING_POLICY_FINALIZATION", Path(__file__).name,
        "metadata/eda_saturation_qc.csv", "success",
        f"included={included}; possible_saturation_included={possible}; "
        f"saturated_excluded={saturated}; possible classification preserved=true; "
        "exclusion_scope=eda_only; raw_files_modified=false",
    )
    print(f"EDA policy: included={included}/{len(rows)}, possible retained={possible}, "
          f"saturated excluded={saturated}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
