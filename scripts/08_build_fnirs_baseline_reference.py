#!/usr/bin/env python3
"""Build participant-specific fNIRS baseline-current references."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from multimodalphysiokit.io import read_biosignalsplux_hdf5

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dataset.provenance import log_processing  # noqa: E402
from processing.common import participant_metadata_from_xlsx, read_csv_rows  # noqa: E402
from processing.fnirs_reference import build_fnirs_reference  # noqa: E402

FIELDS = (
    "participant_id", "age_years", "age_source", "baseline_hdf5",
    "first_event_sample", "first_event_time_s", "reference_start_sample",
    "reference_end_sample", "reference_start_time_s", "reference_end_time_s",
    "reference_duration_s", "sampling_rate", "red_reference", "infrared_reference",
    "reference_method", "dpf_660nm", "dpf_860nm", "fnirs_reference_status",
)


def plot_reference(row: dict[str, object], *, show: bool, save: bool) -> None:
    if not row["first_event_sample"]:
        return
    source = ROOT / str(row["baseline_hdf5"])
    rec = read_biosignalsplux_hdf5(source)
    red, infrared = rec.get_signal("fnirs_red"), rec.get_signal("fnirs_infrared")
    fs = float(row["sampling_rate"]); time = np.arange(red.n_samples) / fs
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(15, 7), constrained_layout=True)
    for axis, signal, label in zip(axes, (red, infrared), ("fNIRS Red", "fNIRS Infrared")):
        axis.plot(time, signal.samples, linewidth=.55)
        axis.axvline(float(row["first_event_time_s"]), color="black", linestyle="--", label="First rising edge")
        axis.axvline(float(row["first_event_time_s"])+210, color="tab:orange", linestyle=":", label="Target +210 s")
        if row["reference_start_time_s"] != "":
            axis.axvline(float(row["reference_start_time_s"]), color="tab:green", linestyle="--", label="Reference start (+30 s)")
            axis.axvline(float(row["reference_end_time_s"]), color="tab:red", linestyle="--", label="Reference end (+210 s)")
            axis.axvspan(float(row["reference_start_time_s"]), float(row["reference_end_time_s"]), color="tab:green", alpha=.12)
        axis.set_ylabel(f"{label} [{signal.units}]")
    axes[0].legend(loc="best"); axes[-1].set_xlabel("Time from baseline recording start [s]")
    fig.suptitle(f"{row['participant_id']} — age {row['age_years'] or 'missing'} — first event {float(row['first_event_time_s']):.3f}s\n"
                 f"reference {row['reference_start_time_s']}–{row['reference_end_time_s']}s — {row['fnirs_reference_status']}")
    if save:
        output = ROOT / "outputs/qc/fnirs_baseline_reference" / f"{row['participant_id']}_fnirs_reference.png"
        output.parent.mkdir(parents=True, exist_ok=True); fig.savefig(output, dpi=180)
    if show: plt.show()
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true"); group.add_argument("--subject", type=int)
    parser.add_argument("--show-plots", action="store_true"); parser.add_argument("--save-plots", action="store_true")
    args = parser.parse_args()
    if args.all and args.show_plots: parser.error("--show-plots is only available with --subject")

    participants = participant_metadata_from_xlsx(ROOT / "data/SecondRun_Professional.xlsx")
    if len(participants) != 21: raise ValueError(f"Expected 21 participant rows, found {len(participants)}")
    ids = [r["participant_id"] for r in participants]
    if len(ids) != len(set(ids)): raise ValueError("Duplicate participant mapping")
    physiology = {(r["participant_id"], r["phase"]): r for r in read_csv_rows(ROOT / "metadata/physiological_recordings.csv")}
    selected = participants if args.all else [r for r in participants if r["participant_id"] == f"Subject_{args.subject}"]
    if len(selected) != (21 if args.all else 1): raise ValueError("Participant selection failed")
    rows = []
    for participant in selected:
        pid = participant["participant_id"]; age_text = participant["age_years"]
        age = float(age_text) if age_text else None
        source = physiology[(pid, "baseline")]["physio_filepath"]
        row = build_fnirs_reference(pid, age, ROOT / source, source)
        rows.append(row); plot_reference(row, show=args.show_plots, save=args.save_plots)

    print("participant_id,age_years,first_event_time_s,reference_start_time_s,reference_end_time_s,reference_duration_s,red_reference,infrared_reference,dpf_660nm,dpf_860nm,fnirs_reference_status")
    for r in rows:
        print(",".join(str(r[k]) for k in ("participant_id","age_years","first_event_time_s","reference_start_time_s","reference_end_time_s","reference_duration_s","red_reference","infrared_reference","dpf_660nm","dpf_860nm","fnirs_reference_status")))
    if args.all:
        path = ROOT / "metadata/fnirs_baseline_reference.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS); writer.writeheader(); writer.writerows(rows)
        statuses = {status: sum(r["fnirs_reference_status"] == status for r in rows) for status in sorted({str(r["fnirs_reference_status"]) for r in rows})}
        log_processing(ROOT/"metadata/processing_log.csv", "FNIRS_BASELINE_REFERENCE_EXTRACTION", Path(__file__).name,
                       "metadata/fnirs_baseline_reference.csv", "success", f"rows=21; statuses={statuses}; anchor=first_rising_edge+30s; duration=180s; raw_files_modified=false")
    return 0


if __name__ == "__main__": raise SystemExit(main())
