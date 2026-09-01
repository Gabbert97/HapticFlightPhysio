#!/usr/bin/env python3
"""Classify participant ECG polarity and create pending manual-review plots.

This is a QC-only workflow. Temporary band-pass filtering is used solely for
polarity-invariant QRS detection, polarity estimation, and objective window
selection. Figures always show the original MultimodalPhysioKit-converted ECG.

Window selection rule
---------------------
Every complete 30-second window on a 30-second grid, with a preferred 10-second
recording-edge margin, is assessed. The selected window maximizes a quality
score that does not include polarity: plausible beat count (20--60), fraction
of RR intervals corresponding to 40--160 bpm, RR regularity, and penalties for
repeated extrema (clipping), flat differences, and robust extreme amplitudes.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from multimodalphysiokit.core import Signal
from multimodalphysiokit.io import read_biosignalsplux_hdf5
from scipy.signal import butter, find_peaks, sosfiltfilt

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from dataset.naming import CANONICAL_PHASES  # noqa: E402
from dataset.provenance import log_processing  # noqa: E402

SCRIPT_NAME = Path(__file__).name
WINDOW_SECONDS = 30.0
EDGE_MARGIN_SECONDS = 10.0
MIN_CONFIDENCE = 0.65
MIN_ABSOLUTE_POLARITY_SCORE = 0.50
TOP_WINDOWS_FOR_PARTICIPANT = 5

FIELDS = [
    "participant_id", "participant_numeric_id", "ecg_polarity_auto",
    "polarity_confidence", "participant_polarity_score", "ecg_multiplier_auto",
    "ecg_polarity_manual", "ecg_multiplier_manual", "ecg_polarity_final",
    "ecg_multiplier_final", "manual_review_required", "manual_review_status",
    "qc_review_phase", "qc_review_start_sample", "qc_review_end_sample",
    "qc_review_start_time_s", "qc_review_end_time_s", "qc_review_n_beats",
    "qc_review_plot", "qc_window_quality_score", "qc_rr_plausible_fraction",
    "qc_rr_robust_cv", "qc_clipping_fraction", "qc_flat_difference_fraction",
    "qc_extreme_amplitude_fraction", "classification_method", "notes",
]


@dataclass
class WindowAssessment:
    phase: str
    source_path: Path
    start_sample: int
    end_sample: int
    sampling_rate: float
    raw_ecg: np.ndarray
    qrs_indices_local: np.ndarray
    quality_score: float
    polarity_score: float
    n_beats: int
    rr_plausible_fraction: float
    rr_robust_cv: float
    clipping_fraction: float
    flat_difference_fraction: float
    extreme_amplitude_fraction: float


def read_physio_index(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"participant_id", "participant_numeric_id", "phase", "physio_filepath",
                "sampling_rate_hz", "n_samples"}
    missing = required - set(rows[0] if rows else [])
    if missing:
        raise ValueError(f"Missing physiological index columns: {sorted(missing)}")
    return rows


def candidate_starts(n_samples: int, sampling_rate: float) -> list[int]:
    window = int(round(WINDOW_SECONDS * sampling_rate))
    margin = int(round(EDGE_MARGIN_SECONDS * sampling_rate))
    if n_samples < window:
        return []
    latest = n_samples - margin - window
    if latest >= margin:
        return list(range(margin, latest + 1, window))
    return [(n_samples - window) // 2]


def assess_window(raw: np.ndarray, sampling_rate: float, phase: str,
                  source_path: Path, start_sample: int) -> WindowAssessment:
    if raw.size != int(round(WINDOW_SECONDS * sampling_rate)):
        raise ValueError("QC candidate is not exactly 30 seconds")
    centered = raw - np.median(raw)
    sos = butter(3, [5.0, 25.0], btype="bandpass", fs=sampling_rate, output="sos")
    filtered = sosfiltfilt(sos, centered)

    envelope = np.abs(filtered)
    envelope_median = float(np.median(envelope))
    envelope_mad = float(np.median(np.abs(envelope - envelope_median)))
    threshold = envelope_median + 3.0 * envelope_mad
    candidates, _ = find_peaks(
        envelope, height=threshold, distance=int(round(0.30 * sampling_rate))
    )
    radius = int(round(0.08 * sampling_rate))
    refined = []
    for candidate in candidates:
        left, right = max(0, candidate - radius), min(raw.size, candidate + radius + 1)
        refined.append(left + int(np.argmax(np.abs(filtered[left:right]))))
    peaks = np.unique(np.asarray(refined, dtype=int))

    rr = np.diff(peaks) / sampling_rate
    plausible = (rr >= 60.0 / 160.0) & (rr <= 60.0 / 40.0)
    rr_fraction = float(np.mean(plausible)) if rr.size else 0.0
    if np.any(plausible):
        plausible_rr = rr[plausible]
        median_rr = float(np.median(plausible_rr))
        rr_robust_cv = float(np.median(np.abs(plausible_rr - median_rr)) / median_rr)
    else:
        rr_robust_cv = 1.0

    scale = float(np.median(np.abs(raw - np.median(raw))))
    scale = max(scale, np.finfo(float).eps)
    clipping = float(max(np.mean(raw == np.min(raw)), np.mean(raw == np.max(raw))))
    flat = float(np.mean(np.abs(np.diff(raw)) <= max(scale * 1e-5, 1e-12)))
    extreme = float(np.mean(np.abs(raw - np.median(raw)) > 12.0 * scale))
    beat_count_score = min(peaks.size / 20.0, 1.0) * min(60.0 / max(peaks.size, 1), 1.0)
    quality = (
        beat_count_score
        * rr_fraction
        * math.exp(-3.0 * rr_robust_cv)
        * max(0.0, 1.0 - 20.0 * clipping)
        * max(0.0, 1.0 - 2.0 * flat)
        * max(0.0, 1.0 - 5.0 * extreme)
    )
    peak_amplitudes = filtered[peaks] if peaks.size else np.asarray([], dtype=float)
    polarity = (
        float(np.median(peak_amplitudes) / (np.median(np.abs(peak_amplitudes)) + 1e-12))
        if peak_amplitudes.size else 0.0
    )
    return WindowAssessment(
        phase, source_path, start_sample, start_sample + raw.size, sampling_rate,
        raw.copy(), peaks, float(quality), polarity, int(peaks.size), rr_fraction,
        rr_robust_cv, clipping, flat, extreme,
    )


def assess_participant(rows: list[dict[str, str]]) -> tuple[str, float, float, WindowAssessment]:
    assessments: list[WindowAssessment] = []
    for row in sorted(rows, key=lambda item: CANONICAL_PHASES.index(item["phase"])):
        path = REPOSITORY_ROOT / row["physio_filepath"]
        recording = read_biosignalsplux_hdf5(path)
        ecg = recording.get_signal("ecg")
        expected_fs, expected_n = float(row["sampling_rate_hz"]), int(row["n_samples"])
        if ecg.sampling_frequency != expected_fs or ecg.n_samples != expected_n:
            raise ValueError(f"ECG timing disagreement for {path}")
        for start in candidate_starts(ecg.n_samples, ecg.sampling_frequency):
            stop = start + int(round(WINDOW_SECONDS * ecg.sampling_frequency))
            assessments.append(assess_window(
                ecg.samples[start:stop], ecg.sampling_frequency, row["phase"], path, start
            ))
    if not assessments:
        raise ValueError("No complete 30-second ECG window exists")
    ranked = sorted(assessments, key=lambda item: item.quality_score, reverse=True)
    representative = ranked[0]
    top = ranked[:TOP_WINDOWS_FOR_PARTICIPANT]
    weights = np.asarray([max(item.quality_score, 1e-9) for item in top])
    scores = np.asarray([item.polarity_score for item in top])
    participant_score = float(np.average(scores, weights=weights))
    dominant_sign = 1 if participant_score >= 0 else -1
    agreement = float(np.average(np.sign(scores) == dominant_sign, weights=weights))
    rhythm = float(np.average([
        item.rr_plausible_fraction * math.exp(-3.0 * item.rr_robust_cv) for item in top
    ], weights=weights))
    confidence = float(np.clip(abs(participant_score) * agreement * rhythm, 0.0, 1.0))
    if confidence < MIN_CONFIDENCE or abs(participant_score) < MIN_ABSOLUTE_POLARITY_SCORE:
        classification = "uncertain"
    else:
        classification = "normal" if participant_score > 0 else "inverted"
    return classification, confidence, participant_score, representative


def plot_review(participant_id: str, classification: str, confidence: float,
                assessment: WindowAssessment, output_path: Path,
                show_qrs_markers: bool, overwrite: bool) -> None:
    if output_path.exists() and not overwrite:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    time = np.arange(assessment.raw_ecg.size) / assessment.sampling_rate
    fig, axis = plt.subplots(figsize=(15, 6), constrained_layout=True)
    axis.plot(time, assessment.raw_ecg, color="tab:blue", linewidth=0.75)
    if show_qrs_markers and assessment.qrs_indices_local.size:
        indexes = assessment.qrs_indices_local
        axis.scatter(time[indexes], assessment.raw_ecg[indexes], s=12, color="black",
                     alpha=0.65, zorder=3, label="QC QRS locations")
        axis.legend(loc="upper right")
    proposed = "-1 (pending manual review)" if classification == "inverted" else "unresolved"
    subject = participant_id.replace("_", " ")
    axis.set_title(
        f"{subject} — ECG polarity QC\n"
        f"Source: {assessment.phase} | Automatic classification: {classification.upper()} | "
        f"Confidence: {confidence:.3f} | Proposed multiplier: {proposed}"
    )
    axis.set_xlabel("Time within selected window [s]")
    axis.set_ylabel("ECG [mV]")
    axis.set_xlim(0.0, WINDOW_SECONDS)
    axis.grid(True, alpha=0.2)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def existing_manual_values(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {row["participant_id"]: row for row in csv.DictReader(handle)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--show-qrs-markers", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--metadata-dir", type=Path, default=REPOSITORY_ROOT / "metadata")
    parser.add_argument("--output-dir", type=Path,
                        default=REPOSITORY_ROOT / "outputs/qc/ecg_polarity/manual_review")
    args = parser.parse_args()

    physio_rows = read_physio_index(args.metadata_dir / "physiological_recordings.csv")
    qc_path = args.metadata_dir / "ecg_polarity_qc.csv"
    previous = existing_manual_values(qc_path)
    participants = sorted({row["participant_id"] for row in physio_rows},
                          key=lambda value: int(value.split("_")[-1]))
    output_rows = []
    review_table = []
    for participant_id in participants:
        selected_rows = [row for row in physio_rows if row["participant_id"] == participant_id]
        classification, confidence, score, review = assess_participant(selected_rows)
        review_required = classification in {"inverted", "uncertain"}
        old = previous.get(participant_id, {})
        manual_polarity = old.get("ecg_polarity_manual", "")
        manual_multiplier = old.get("ecg_multiplier_manual", "")
        reviewed = bool(manual_polarity and manual_multiplier)
        if review_required:
            review_status = "reviewed" if reviewed else "pending"
            final_polarity = manual_polarity if reviewed else ""
            final_multiplier = manual_multiplier if reviewed else ""
        else:
            review_status = "not_required"
            final_polarity, final_multiplier = "normal", "1"
        auto_multiplier = "1" if classification == "normal" else (
            "-1" if classification == "inverted" else ""
        )
        plot_relative = ""
        if review_required:
            plot_path = args.output_dir / f"{participant_id}_ecg_polarity_qc.png"
            plot_review(participant_id, classification, confidence, review, plot_path,
                        args.show_qrs_markers, args.overwrite)
            plot_relative = plot_path.resolve().relative_to(REPOSITORY_ROOT.resolve()).as_posix()
            review_table.append((participant_id, classification, confidence, auto_multiplier or "unresolved",
                                 review.phase, review.start_sample / review.sampling_rate,
                                 review.end_sample / review.sampling_rate, review.n_beats,
                                 plot_relative, review_status))
        output_rows.append({
            "participant_id": participant_id,
            "participant_numeric_id": int(participant_id.split("_")[-1]),
            "ecg_polarity_auto": classification,
            "polarity_confidence": f"{confidence:.6f}",
            "participant_polarity_score": f"{score:.6f}",
            "ecg_multiplier_auto": auto_multiplier,
            "ecg_polarity_manual": manual_polarity,
            "ecg_multiplier_manual": manual_multiplier,
            "ecg_polarity_final": final_polarity,
            "ecg_multiplier_final": final_multiplier,
            "manual_review_required": str(review_required),
            "manual_review_status": review_status,
            "qc_review_phase": review.phase if review_required else "",
            "qc_review_start_sample": review.start_sample if review_required else "",
            "qc_review_end_sample": review.end_sample if review_required else "",
            "qc_review_start_time_s": f"{review.start_sample / review.sampling_rate:.6f}" if review_required else "",
            "qc_review_end_time_s": f"{review.end_sample / review.sampling_rate:.6f}" if review_required else "",
            "qc_review_n_beats": review.n_beats if review_required else "",
            "qc_review_plot": plot_relative,
            "qc_window_quality_score": f"{review.quality_score:.6f}" if review_required else "",
            "qc_rr_plausible_fraction": f"{review.rr_plausible_fraction:.6f}" if review_required else "",
            "qc_rr_robust_cv": f"{review.rr_robust_cv:.6f}" if review_required else "",
            "qc_clipping_fraction": f"{review.clipping_fraction:.6f}" if review_required else "",
            "qc_flat_difference_fraction": f"{review.flat_difference_fraction:.6f}" if review_required else "",
            "qc_extreme_amplitude_fraction": f"{review.extreme_amplitude_fraction:.6f}" if review_required else "",
            "classification_method": "polarity-invariant absolute bandpass-QRS; top-5 quality windows",
            "notes": "No ECG correction applied; pending rows require manual review" if review_required else "",
        })
        print(f"{participant_id}: {classification}; confidence={confidence:.3f}; "
              f"score={score:.3f}; review={review_status}")

    with qc_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(output_rows)
    counts = {name: sum(row["ecg_polarity_auto"] == name for row in output_rows)
              for name in ("normal", "inverted", "uncertain")}
    log_processing(
        args.metadata_dir / "processing_log.csv", "ECG_POLARITY_MANUAL_REVIEW_PREPARATION",
        SCRIPT_NAME, "metadata/ecg_polarity_qc.csv", "success",
        f"normal={counts['normal']}; inverted={counts['inverted']}; uncertain={counts['uncertain']}; "
        f"manual_review_plots={len(review_table)}; window_seconds=30; correction_applied=false",
    )
    print("\nManual-review table")
    print("participant_id | automatic_classification | confidence | proposed_multiplier | "
          "selected_phase | window_start_s | window_end_s | n_beats | plot_filepath | status")
    for row in review_table:
        print(" | ".join(str(value) if not isinstance(value, float) else f"{value:.3f}" for value in row))
    print(f"\nCounts: {counts}; manual-review plots={len(review_table)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
