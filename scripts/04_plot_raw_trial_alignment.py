#!/usr/bin/env python3
"""Plot complete converted HDF5 recordings with validated boundary markers."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from multimodalphysiokit.io import read_biosignalsplux_hdf5

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from dataset.baseline_events import read_baseline_rising_edges  # noqa: E402
from dataset.naming import CANONICAL_PHASES  # noqa: E402
from dataset.provenance import log_processing  # noqa: E402

SIGNALS = (
    ("ecg", "ECG"),
    ("eda", "EDA"),
    ("rip", "RESP"),
    ("temp", "TEMP"),
    ("fnirs_red", "fNIRS Red"),
    ("fnirs_infrared", "fNIRS Infrared"),
)
TRIAL_REQUIRED_COLUMNS = {
    "participant_id", "phase", "trial_number_chronological", "physio_start_sample",
    "physio_end_sample", "include_physio_analysis", "alignment_status",
}
PHYSIO_REQUIRED_COLUMNS = {
    "participant_id", "phase", "physio_filepath", "sampling_rate_hz", "n_samples",
}
BASELINE_FIELDS = [
    "participant_id", "phase", "event_number", "event_sample", "event_time_s",
    "digital_previous_value", "digital_new_value", "source_hdf5", "source_dataset",
    "sampling_rate_hz",
]
MANIFEST_FIELDS = [
    "participant_id", "phase", "plot_filepath", "status", "boundary_source",
    "n_boundaries", "n_events", "event_times_s", "n_segments", "segment_labels",
    "n_valid_trials", "signals_plotted", "signal_units",
]


def read_csv(path: Path, required: set[str]) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Metadata file not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
        return list(reader)


def write_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def normalize_phase(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]", "", value.lower())
    phases = {re.sub(r"[^a-z0-9]", "", phase): phase for phase in CANONICAL_PHASES}
    try:
        return phases[normalized]
    except KeyError as exc:
        raise ValueError(f"Unsupported phase: {value}") from exc


def parse_bool(value: str) -> bool:
    if value.strip().lower() == "true":
        return True
    if value.strip().lower() == "false":
        return False
    raise ValueError(f"Expected True/False metadata value, got {value!r}")


def repository_relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPOSITORY_ROOT.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def extract_all_baseline_events(
    physio_rows: list[dict[str, str]], metadata_dir: Path
) -> tuple[dict[tuple[str, str], list[object]], list[dict[str, object]]]:
    by_recording: dict[tuple[str, str], list[object]] = {}
    index_rows: list[dict[str, object]] = []
    baseline_rows = sorted(
        (row for row in physio_rows if row["phase"] == "baseline"),
        key=lambda row: int(row["participant_id"].split("_")[-1]),
    )
    for row in baseline_rows:
        path = REPOSITORY_ROOT / row["physio_filepath"]
        events, sampling_rate, n_samples = read_baseline_rising_edges(path)
        if sampling_rate != float(row["sampling_rate_hz"]) or n_samples != int(row["n_samples"]):
            raise ValueError(f"Baseline timing metadata disagreement for {path}")
        by_recording[(row["participant_id"], "baseline")] = events
        for event in events:
            index_rows.append({
                "participant_id": row["participant_id"],
                "phase": "baseline",
                "event_number": event.event_number,
                "event_sample": event.event_sample,
                "event_time_s": f"{event.event_time_s:.6f}",
                "digital_previous_value": event.digital_previous_value,
                "digital_new_value": event.digital_new_value,
                "source_hdf5": row["physio_filepath"],
                "source_dataset": event.source_dataset,
                "sampling_rate_hz": f"{sampling_rate:g}",
            })
    write_csv(metadata_dir / "baseline_event_index.csv", BASELINE_FIELDS, index_rows)
    return by_recording, index_rows


def validate_trials(rows: list[dict[str, str]], sampling_rate: float, n_samples: int):
    included, excluded, durations = [], [], []
    for row in sorted(rows, key=lambda item: int(item["trial_number_chronological"])):
        target = included if parse_bool(row["include_physio_analysis"]) else excluded
        target.append(row)
        if target is excluded:
            continue
        start, end = int(row["physio_start_sample"]), int(row["physio_end_sample"])
        if not 0 <= start < end <= n_samples:
            raise ValueError(f"Invalid included trial bounds for {row}")
        duration = (end - start) / sampling_rate
        if abs(duration - 60.0) > 1.0:
            raise ValueError(f"Included trial is not approximately 60 s: {duration:.6f}")
        durations.append(duration)
    if not included:
        raise ValueError("No trials are eligible for physiological analysis")
    return included, excluded, durations


def load_converted_recording(path: Path, expected_rate: float, expected_samples: int):
    recording = read_biosignalsplux_hdf5(path)
    for signal_name, _ in SIGNALS:
        if not recording.has_signal(signal_name):
            raise ValueError(f"Missing required signal {signal_name!r} in {path}")
        signal = recording.get_signal(signal_name)
        if signal.n_samples != expected_samples or signal.sampling_frequency != expected_rate:
            raise ValueError(
                f"Converted signal timing differs from HDF5 metadata for {path}/{signal_name}"
            )
    return recording


def make_plot(
    recording, participant_id: str, phase: str, included: list[dict[str, str]],
    excluded: list[dict[str, str]], events: list[object], show_excluded: bool,
):
    fs = recording.get_signal("ecg").sampling_frequency
    figure, axes = plt.subplots(
        len(SIGNALS), 1, figsize=(16, 14), sharex=True, constrained_layout=True
    )
    for axis, (signal_name, display_name) in zip(axes, SIGNALS, strict=True):
        signal = recording.get_signal(signal_name)
        axis.plot(signal.time, signal.samples, linewidth=0.45, color="tab:blue", rasterized=True)
        axis.set_ylabel(f"{display_name}\n[{signal.units}]")
        axis.grid(True, alpha=0.2)
        if phase == "baseline":
            for event in events:
                axis.axvline(event.event_time_s, color="black", linestyle="--", linewidth=0.8)
        else:
            for row in included:
                for field in ("physio_start_sample", "physio_end_sample"):
                    axis.axvline(int(row[field]) / fs, color="black", linestyle="--", linewidth=0.8)
            if show_excluded:
                for row in excluded:
                    for field in ("physio_start_sample", "physio_end_sample"):
                        if row[field]:
                            axis.axvline(int(row[field]) / fs, color="red", linestyle=":", linewidth=1.0)

    if phase == "baseline":
        for event in events:
            axes[0].text(event.event_time_s, 0.98, event.event_number,
                         transform=axes[0].get_xaxis_transform(), ha="left", va="top", fontsize=8)
    else:
        for row in included:
            axes[0].text(int(row["physio_start_sample"]) / fs, 0.98,
                         f"T{row['trial_number_chronological']}",
                         transform=axes[0].get_xaxis_transform(), ha="left", va="top", fontsize=8)
        if show_excluded:
            for row in excluded:
                if row["physio_start_sample"]:
                    axes[0].text(int(row["physio_start_sample"]) / fs, 0.86,
                                 f"EXCL T{row['trial_number_chronological']}", color="red",
                                 transform=axes[0].get_xaxis_transform(), ha="left", va="top", fontsize=7)

    axes[-1].set_xlim(0, recording.get_signal("ecg").n_samples / fs)
    axes[-1].set_xlabel("Time from physiological recording start [s]")
    title = phase.replace("_", " ").title()
    figure.suptitle(f"{participant_id.replace('_', ' ')} — {title} — Raw physiological signals")
    return figure


def plot_one(
    physio: dict[str, str], trial_rows: list[dict[str, str]], baseline_events: list[object],
    output_dir: Path, save: bool, overwrite: bool, show_excluded: bool,
) -> dict[str, object]:
    participant_id, phase = physio["participant_id"], physio["phase"]
    source = REPOSITORY_ROOT / physio["physio_filepath"]
    fs, n_samples = float(physio["sampling_rate_hz"]), int(physio["n_samples"])
    included: list[dict[str, str]] = []
    excluded: list[dict[str, str]] = []
    durations: list[float] = []
    if phase != "baseline":
        selected = [r for r in trial_rows if r["participant_id"] == participant_id and r["phase"] == phase]
        included, excluded, durations = validate_trials(selected, fs, n_samples)
    recording = load_converted_recording(source, fs, n_samples)
    figure = make_plot(recording, participant_id, phase, included, excluded, baseline_events, show_excluded)

    output_path = output_dir / participant_id / f"{phase}.png"
    status = "displayed"
    if save:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        existed_before = output_path.exists()
        if existed_before and not overwrite:
            status = "skipped_existing"
        else:
            figure.savefig(output_path, dpi=140, bbox_inches="tight")
            status = "overwritten" if existed_before else "generated"
        plt.close(figure)
    else:
        plt.show()

    if phase == "baseline":
        labels = [event.event_number for event in baseline_events]
        event_times = [round(event.event_time_s, 6) for event in baseline_events]
        boundary_source, n_boundaries, n_events, n_segments, n_valid = (
            "hdf5_digital_rising_edge", len(baseline_events), len(baseline_events), 0, 0
        )
    else:
        labels = [f"T{row['trial_number_chronological']}" for row in included]
        event_times = []
        boundary_source, n_boundaries, n_events, n_segments, n_valid = (
            "behavioral_timestamp_alignment", 2 * len(included), 0, len(included), len(included)
        )
    print(f"{participant_id}/{phase}: {status}; {boundary_source}; labels={','.join(labels)}")
    if durations:
        print(f"  included duration range: {min(durations):.6f}–{max(durations):.6f} s")
    return {
        "participant_id": participant_id, "phase": phase,
        "plot_filepath": repository_relative(output_path) if save else "",
        "status": status, "boundary_source": boundary_source,
        "n_boundaries": n_boundaries, "n_events": n_events,
        "event_times_s": json.dumps(event_times, separators=(",", ":")),
        "n_segments": n_segments, "segment_labels": ";".join(labels),
        "n_valid_trials": n_valid,
        "signals_plotted": ";".join(display_name for _, display_name in SIGNALS),
        "signal_units": ";".join(
            f"{display_name}={recording.get_signal(signal_name).units}"
            for signal_name, display_name in SIGNALS
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--all", action="store_true", help="save every available participant/phase plot")
    selection.add_argument("--subject", type=int)
    parser.add_argument("--phase", help="required with --subject")
    parser.add_argument("--show-excluded", action="store_true")
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--metadata-dir", type=Path, default=REPOSITORY_ROOT / "metadata")
    parser.add_argument("--output-dir", type=Path, default=REPOSITORY_ROOT / "outputs" / "alignment_plots")
    args = parser.parse_args()
    if not args.all and not args.phase:
        parser.error("--phase is required with --subject")

    physio_rows = read_csv(args.metadata_dir / "physiological_recordings.csv", PHYSIO_REQUIRED_COLUMNS)
    trial_rows = read_csv(args.metadata_dir / "trial_index.csv", TRIAL_REQUIRED_COLUMNS)
    event_map, event_index = extract_all_baseline_events(physio_rows, args.metadata_dir)
    counts = {}
    for row in (r for r in physio_rows if r["phase"] == "baseline"):
        counts[row["participant_id"]] = len(event_map[(row["participant_id"], "baseline")])
    log_processing(
        args.metadata_dir / "processing_log.csv", "BASELINE_DIGITAL_EVENT_EXTRACTION",
        Path(__file__).name, "data/Subject_*/raw_bp_signals/baseline.h5", "success",
        f"rule=0->1; total_events={len(event_index)}; participants_1_event={sum(v == 1 for v in counts.values())}; "
        f"participants_2_events={sum(v == 2 for v in counts.values())}; "
        f"other_counts={json.dumps({k: v for k, v in counts.items() if v not in (1, 2)}, sort_keys=True)}",
    )

    if args.all:
        selected = sorted(
            physio_rows,
            key=lambda row: (int(row["participant_id"].split("_")[-1]), CANONICAL_PHASES.index(row["phase"])),
        )
        save = True
    else:
        phase = normalize_phase(args.phase)
        participant_id = f"Subject_{args.subject}"
        selected = [r for r in physio_rows if r["participant_id"] == participant_id and r["phase"] == phase]
        if len(selected) != 1:
            raise ValueError(f"Expected one recording for {participant_id}/{phase}, found {len(selected)}")
        save = args.save

    manifest = []
    for physio in selected:
        manifest.append(plot_one(
            physio, trial_rows, event_map.get((physio["participant_id"], "baseline"), []),
            args.output_dir, save, args.overwrite, args.show_excluded,
        ))
    if args.all:
        write_csv(args.metadata_dir / "alignment_plot_manifest.csv", MANIFEST_FIELDS, manifest)
        generated = sum(row["status"] in {"generated", "overwritten"} for row in manifest)
        skipped = sum(row["status"] == "skipped_existing" for row in manifest)
        log_processing(
            args.metadata_dir / "processing_log.csv", "ALIGNMENT_PLOT_BATCH_GENERATION",
            Path(__file__).name, repository_relative(args.output_dir), "success",
            f"recordings={len(manifest)}; generated_or_overwritten={generated}; skipped_existing={skipped}; "
            "baseline=hdf5_digital_rising_edge; experimental=behavioral_timestamp_alignment",
        )
        log_processing(
            args.metadata_dir / "processing_log.csv", "FNIRS_RAW_CURRENT_PLOT_INTEGRATION",
            Path(__file__).name, repository_relative(args.output_dir), "success",
            "signals=fnirs_red,fnirs_infrared; conversion=MultimodalPhysioKit ADC-to-uA; "
            "preprocessing=none; alignment=existing shared-acquisition sample indices; "
            f"recordings={len(manifest)}",
        )
        print(f"Batch complete: {len(manifest)} recordings; generated/overwritten={generated}; skipped={skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
