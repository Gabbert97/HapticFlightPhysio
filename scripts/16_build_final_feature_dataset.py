#!/usr/bin/env python3
"""Assemble the validated physiological features into one analysis-ready table."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/final_features"
PHASE_ORDER = {name: index for index, name in enumerate(
    ("baseline", "pre_test", "test_1", "test_2", "test_3", "evaluation")
)}
SOURCES = {
    "ecg": (
        ROOT / "outputs/features/ecg/baseline_features.csv",
        ROOT / "outputs/features/ecg/experimental_features.csv",
    ),
    "eda": (
        ROOT / "outputs/features/eda/baseline_features.csv",
        ROOT / "outputs/features/eda/experimental_features.csv",
    ),
    "resp": (
        ROOT / "outputs/features/resp/rate_3_30/baseline_features.csv",
        ROOT / "outputs/features/resp/rate_3_30/experimental_features.csv",
    ),
    "temp": (
        ROOT / "outputs/features/temp/baseline_features.csv",
        ROOT / "outputs/features/temp/experimental_features.csv",
    ),
    "fnirs": (
        ROOT / "outputs/features/fnirs/baseline_features.csv",
        ROOT / "outputs/features/fnirs/experimental_features.csv",
    ),
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def truth(value: str) -> bool:
    return value.strip().lower() == "true"


def trial_value(value: str) -> int | None:
    return None if not value.strip() else int(float(value))


def key(row: dict[str, str]) -> tuple[str, str, int | None]:
    return row["participant_id"], row["phase"], trial_value(row.get("trial_number", ""))


def feature_name(modality: str, source_name: str) -> str:
    if modality == "resp":
        return source_name.replace("respiration_", "resp_", 1)
    if modality == "temp":
        return source_name.replace("temperature_", "temp_", 1)
    return source_name


def is_feature(modality: str, name: str) -> bool:
    if name in {"fnirs_reference_start_sample", "fnirs_reference_end_sample"}:
        return False
    prefixes = {
        "ecg": "ecg_", "eda": "eda_", "resp": "respiration_",
        "temp": "temperature_", "fnirs": "fnirs_",
    }
    return name.startswith(prefixes[modality])


def description_and_unit(name: str, modality: str) -> tuple[str, str]:
    exact = {
        "ecg_bpm_mean": ("Mean heart rate across accepted beats.", "beats/min"),
        "ecg_bpm_median": ("Median heart rate across accepted beats.", "beats/min"),
        "ecg_bpm_std": ("Standard deviation of heart rate across accepted beats.", "beats/min"),
        "ecg_pnn50": ("Percentage of successive NN intervals differing by more than 50 ms.", "%"),
        "ecg_pnn50_max": ("Maximum package pNN50 estimate.", "%"),
        "ecg_plf_mean": ("Mean low-frequency HRV power returned by frequency mode 2.", ""),
        "ecg_plf_std": ("Standard deviation of low-frequency HRV power returned by frequency mode 2.", ""),
        "ecg_phf_mean": ("Mean high-frequency HRV power returned by frequency mode 2.", ""),
        "ecg_phf_std": ("Standard deviation of high-frequency HRV power returned by frequency mode 2.", ""),
        "ecg_lf_hf_mean": ("Mean LF/HF power ratio returned by frequency mode 2.", "unitless"),
        "ecg_lf_hf_std": ("Standard deviation of the LF/HF power ratio.", "unitless"),
        "eda_scl_mean": ("Mean tonic skin-conductance level.", "µS"),
        "eda_scl_std": ("Standard deviation of tonic skin-conductance level.", "µS"),
        "eda_scl_slope": ("Linear slope of tonic skin-conductance level.", "µS/s"),
        "eda_scr_amplitude_mean": ("Mean accepted skin-conductance response amplitude.", "µS"),
        "eda_scr_amplitude_std": ("Standard deviation of accepted SCR amplitudes.", "µS"),
        "eda_scr_rise_time_mean": ("Mean accepted SCR rise time.", "s"),
        "eda_scr_rise_time_std": ("Standard deviation of accepted SCR rise times.", "s"),
        "eda_scr_per_min": ("Accepted skin-conductance responses per minute.", "responses/min"),
        "resp_breath_rate_mean": ("Mean accepted peak-to-peak respiratory rate.", "breaths/min"),
        "resp_breath_rate_std": ("Standard deviation of accepted respiratory rates.", "breaths/min"),
        "resp_inspiratory_time_mean": ("Mean inspiratory time.", "s"),
        "resp_inspiratory_time_std": ("Standard deviation of inspiratory time.", "s"),
        "resp_expiratory_time_mean": ("Mean expiratory time.", "s"),
        "resp_expiratory_time_std": ("Standard deviation of expiratory time.", "s"),
        "resp_timing_ratio_mean": ("Mean inspiratory-to-expiratory timing ratio.", "unitless"),
        "resp_timing_ratio_std": ("Standard deviation of inspiratory-to-expiratory timing ratio.", "unitless"),
        "resp_amplitude_mean": ("Mean RIP tidal-amplitude proxy.", "% full scale"),
        "resp_amplitude_std": ("Standard deviation of the RIP tidal-amplitude proxy.", "% full scale"),
        "resp_minute_ventilation_mean": ("Mean package minute-ventilation proxy.", "% full scale·breaths/min"),
        "resp_minute_ventilation_std": ("Standard deviation of the package minute-ventilation proxy.", "% full scale·breaths/min"),
        "resp_coefficient_of_variation": ("Coefficient of variation of respiratory rate.", "%"),
        "resp_autocorrelation": ("Absolute lag-one correlation of consecutive respiratory rates.", "unitless"),
        "temp_delta": ("End-to-start skin-temperature change.", "°C"),
        "temp_mean": ("Mean skin temperature.", "°C"),
        "temp_std": ("Standard deviation of skin temperature.", "°C"),
        "temp_delta_over_time": ("Temperature change divided by observation duration.", "°C/s"),
        "temp_linear_slope": ("Linear slope of skin temperature.", "°C/s"),
        "temp_derivative_mean": ("Mean first derivative of skin temperature.", "°C/s"),
        "temp_derivative_std": ("Standard deviation of the first temperature derivative.", "°C/s"),
        "temp_derivative_linear_slope": ("Linear slope of the first temperature derivative.", "°C/s²"),
    }
    if name in exact:
        return exact[name]
    channel = "ΔHbO" if "_oxy_" in name else "ΔHbR"
    statistic = name.split("_", 2)[-1].replace("_", " ")
    units = {
        "max frequency": "Hz", "median frequency": "Hz",
        "spectral entropy": "unitless", "entropy": "unitless",
        "skewness": "unitless", "kurtosis": "unitless",
        "polarity": "unitless", "zero crossing": "count",
    }.get(statistic, "")
    return f"MultimodalPhysioKit {statistic} feature of {channel} concentration change.", units


def cleanup_audit() -> str:
    return """Repository cleanup audit (no files deleted)

KEEP
- scripts/01_standardize_hdf5_names.py — canonical HDF5 naming and rename provenance.
- scripts/02_build_trial_index.py — authoritative physiology/behavior alignment index.
- scripts/03_cleanup_empty_behavioral_files.py — reproducible behavioral cleanup provenance.
- scripts/04_plot_raw_trial_alignment.py — visual alignment verification for trials and baseline events.
- scripts/05_ecg_polarity_qc.py — ECG polarity classification and review plots.
- scripts/06_finalize_ecg_polarity.py — authoritative participant ECG multiplier finalization.
- scripts/07_process_physiological_signals.py — primary modality processing entry point.
- scripts/08_build_fnirs_baseline_reference.py — participant age and baseline-reference workflow.
- scripts/eda_saturation_qc.py — EDA saturation classification.
- scripts/finalize_eda_qc_policy.py — approved EDA inclusion policy.
- scripts/14_build_final_feature_outputs.py — authorized baseline intervals, baseline features, and baseline-referenced fNIRS extraction.
- scripts/15_reprocess_resp_3_30.py — final validated 3–30 breaths/min RESP results.
- scripts/16_build_final_feature_dataset.py — final wide analysis-table assembly.
- src/dataset/ and src/processing/ — reusable discovery, QC, alignment, and processing helpers.
- metadata/*.csv — processing/QC provenance; retain even when superseded for auditability.

ARCHIVE / OPTIONAL
- scripts/09_build_processing_failure_plots.py — useful failure-review tooling, not required for routine final extraction.
- scripts/10_multimodalphysiokit_debug_plots.py — exhaustive package-native visual debugging.
- scripts/12_process_multimodalphysiokit_recordings.py — independent Recording.process_phases validation path.
- scripts/13_validate_ecg_stft.py — mode-1 ECG STFT experiment; production remains frequency mode 2.
- outputs/alignment_plots/ — visual alignment evidence, not an analysis input.
- outputs/multimodalphysiokit_debug/ — package-native trial debug figures.
- outputs/multimodalphysiokit_recordings/ — Recording phase-validation figures.
- outputs/multimodalphysiokit_recording_processing/ — Recording validation processing figures.
- outputs/ecg_stft_validation/ — independent STFT validation artifacts.
- outputs/processing_failures/ and outputs/qc/ — QC evidence and manual-review artifacts.
- outputs/features/baseline_processing_plots/ — native baseline diagnostic plots.
- metadata/recording_* and metadata/multimodalphysiokit_* — independent validation outputs.

REMOVED IN CONSERVATIVE CLEANUP
- scripts/11_flatten_multimodalphysiokit_debug_plots.py — completed one-off directory migration; current output is already flat.
- outputs/ecg_stft_validation/.stage/ — confirmed-empty staging directory, not a scientific result.

No raw data, source feature tables, provenance metadata, or QC decisions are proposed for deletion.
"""


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {OUTPUT}")
    OUTPUT.mkdir(parents=True)

    participants = read_rows(ROOT / "metadata/participant_metadata.csv")
    groups = {row["participant_id"]: row["group"] for row in participants}
    intervals = {
        row["participant_id"]: row
        for row in read_rows(ROOT / "outputs/features/baseline_interval_manifest.csv")
    }
    trials = [
        row for row in read_rows(ROOT / "metadata/trial_index.csv")
        if truth(row["include_physio_analysis"])
    ]

    master: dict[tuple[str, str, int | None], dict[str, object]] = {}
    for participant in participants:
        pid = participant["participant_id"]
        interval = intervals[pid]
        master[(pid, "baseline", None)] = {
            "participant_id": pid, "group": groups[pid], "phase": "baseline",
            "trial_number": "", "baseline_duration_s": interval["baseline_duration_s"],
            "baseline_interval_status": interval["interval_status"],
        }
    for trial in trials:
        observation = (
            trial["participant_id"], trial["phase"],
            int(trial["trial_number_chronological"]),
        )
        if observation in master:
            raise ValueError(f"duplicate validated trial key: {observation}")
        master[observation] = {
            "participant_id": observation[0], "group": groups[observation[0]],
            "phase": observation[1], "trial_number": observation[2],
            "baseline_duration_s": "", "baseline_interval_status": "",
        }

    features_by_modality: dict[str, list[str]] = {}
    successful_keys: dict[str, set[tuple[str, str, int | None]]] = {}
    for modality, paths in SOURCES.items():
        source_rows = read_rows(paths[0]) + read_rows(paths[1])
        feature_columns = sorted({
            feature_name(modality, name)
            for row in source_rows for name in row if is_feature(modality, name)
        })
        features_by_modality[modality] = feature_columns
        successful_keys[modality] = set()
        for row in source_rows:
            if row.get("processing_status", "success") != "success":
                continue
            observation = key(row)
            if observation not in master:
                raise ValueError(f"unexpected {modality} feature key: {observation}")
            if observation in successful_keys[modality]:
                raise ValueError(f"duplicate {modality} feature key: {observation}")
            successful_keys[modality].add(observation)
            for name, value in row.items():
                if is_feature(modality, name):
                    master[observation][feature_name(modality, name)] = value

    ordered = sorted(
        master.values(),
        key=lambda row: (
            int(str(row["participant_id"]).split("_")[1]),
            PHASE_ORDER[str(row["phase"])],
            -1 if row["trial_number"] == "" else int(row["trial_number"]),
        ),
    )
    feature_fields = [name for modality in SOURCES for name in features_by_modality[modality]]
    master_fields = [
        "participant_id", "group", "phase", "trial_number",
        "baseline_duration_s", "baseline_interval_status",
    ] + feature_fields
    write_rows(OUTPUT / "physiological_features_all.csv", ordered, master_fields)

    dictionary = []
    for modality, names in features_by_modality.items():
        for name in names:
            description, unit = description_and_unit(name, modality)
            dictionary.append({
                "feature_name": name, "modality": modality,
                "description": description, "units_if_known": unit,
            })
    write_rows(
        OUTPUT / "feature_dictionary.csv", dictionary,
        ["feature_name", "modality", "description", "units_if_known"],
    )

    expected_by_phase = defaultdict(set)
    for observation in master:
        expected_by_phase[observation[1]].add(observation)
    summary = []
    for modality in SOURCES:
        for phase in PHASE_ORDER:
            expected = expected_by_phase[phase]
            successful = expected & successful_keys[modality]
            summary.append({
                "modality": modality, "phase": phase,
                "expected_rows": len(expected), "successful_rows": len(successful),
                "missing_or_excluded_rows": len(expected - successful),
            })
    write_rows(
        OUTPUT / "processing_summary.csv", summary,
        ["modality", "phase", "expected_rows", "successful_rows", "missing_or_excluded_rows"],
    )

    references = read_rows(ROOT / "outputs/features/fnirs/fnirs_baseline_reference_final.csv")
    reference_rows = [{
        "participant_id": row["participant_id"], "I0_red": row["I0_red"],
        "I0_infrared": row["I0_infrared"],
        "reference_duration_s": row["reference_duration_s"],
        "reference_status": row["reference_status"],
    } for row in references]
    write_rows(
        OUTPUT / "fnirs_baseline_reference.csv", reference_rows,
        ["participant_id", "I0_red", "I0_infrared", "reference_duration_s", "reference_status"],
    )
    (OUTPUT / "repo_cleanup_candidates.txt").write_text(cleanup_audit(), encoding="utf-8")

    expected_count = 21 + len(trials)
    if len(ordered) != expected_count or len({
        (row["participant_id"], row["phase"], row["trial_number"]) for row in ordered
    }) != expected_count:
        raise RuntimeError("final observation-key validation failed")
    print("rows", len(ordered), "baseline", 21, "experimental", len(trials))
    print("features", {name: len(values) for name, values in features_by_modality.items()})
    print("availability", {name: len(values) for name, values in successful_keys.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
