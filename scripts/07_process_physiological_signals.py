#!/usr/bin/env python3
"""Process validated physiological trials with MultimodalPhysioKit 0.1.0."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import traceback
from pathlib import Path

import matplotlib.pyplot as plt
from multimodalphysiokit.io import read_biosignalsplux_hdf5

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from dataset.ecg_polarity import get_ecg_multiplier  # noqa: E402
from dataset.provenance import log_processing  # noqa: E402
from processing.common import (participant_metadata_from_xlsx, read_csv_rows,
                               segment_signal)  # noqa: E402
from processing.ecg import process_ecg  # noqa: E402
from processing.eda import process_eda  # noqa: E402
from processing.fnirs import process_fnirs  # noqa: E402
from processing.resp import process_resp  # noqa: E402
from processing.temp import process_temp  # noqa: E402

MODALITIES = ("ecg", "eda", "resp", "temp", "fnirs")
EXPERIMENTAL_PHASES = ("pre_test", "test_1", "test_2", "test_3", "evaluation")
BASE_FIELDS = ["participant_id", "group", "phase", "trial_number", "source_hdf5",
               "start_sample", "end_sample", "duration_s"]
MANIFEST_FIELDS = [
    "participant_id", "group", "phase", "trial_number", "modality", "included",
    "processing_status", "exclusion_reason", "source_hdf5", "start_sample", "end_sample",
    "sampling_rate_input", "sampling_rate_output", "processing_version",
    "processing_parameters", "output_path", "ecg_multiplier_applied",
    "eda_saturation_status", "eda_processing_decision", "fnirs_processing_stage",
    "fnirs_reference_method", "age_years", "fnirs_reference_start_sample",
    "fnirs_reference_end_sample", "age_source", "red_reference",
    "infrared_reference", "dpf_660nm", "dpf_860nm", "error_message",
]


def bool_value(value: str) -> bool:
    return value.strip().lower() == "true"


def write_dynamic_csv(path: Path, rows: list[dict[str, object]], base_fields: list[str]) -> None:
    feature_fields = sorted({key for row in rows for key in row} - set(base_fields))
    fields = base_fields + feature_fields
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def describe_api() -> None:
    print("MultimodalPhysioKit processing configuration:")
    print("  ECG: ECGProcessor(label_frequency_analysis=2, return_intermediates=True); "
          "4-16 Hz detector filtering, Pan-Tompkins-style QRS, amplitude/IBI screening, "
          "time-domain HR/BPM/pNN50 and PSD LF/HF features")
    print("  EDA: EDAProcessor(minimum_scr_amplitude=0.01); decimate 1000->100 Hz, "
          "5 Hz low-pass, external cvxEDA 1.1.0 defaults tau0=2,tau1=0.7,delta_knot=10,"
          "alpha=8e-4,gamma=1e-2; SCL/SCR features")
    print("  RESP: RespirationProcessor(min_breathing_rate=3.0, "
          "max_breathing_rate=30.0); decimate 1000->100 Hz, fourth-order 1.1 Hz "
          "low-pass, cycle screening and respiratory timing/amplitude features")
    print("  TEMP: TemperatureProcessor(); fifth-order 1 Hz low-pass, package artifact "
          "screening, level/slope/derivative features")
    print("  fNIRS: FNIRSProcessor(age_years=workbook Age) with participant baseline "
          "reference currents, package 1000->10 Hz current filtering, optical density, "
          "age-dependent 660/860nm DPF, MBLL ΔHbO/ΔHbR and package features")


def base_feature(trial: dict[str, str], group: str) -> dict[str, object]:
    start, end, fs = int(trial["physio_start_sample"]), int(trial["physio_end_sample"]), float(trial["sampling_rate_hz"])
    return {"participant_id": trial["participant_id"], "group": group,
            "phase": trial["phase"], "trial_number": trial["trial_number_chronological"],
            "source_hdf5": trial["physio_filepath"], "start_sample": start,
            "end_sample": end, "duration_s": f"{(end-start)/fs:.6f}"}


def output_directory(root: Path, modality: str, trial: dict[str, str]) -> Path:
    return root / modality / trial["participant_id"] / trial["phase"] / f"trial_{int(trial['trial_number_chronological']):02d}"


def process_trial(trial: dict[str, str], selected_modalities: tuple[str, ...],
                  group: str, age_years: float | None, ecg_config: Path,
                  eda_policy: dict[tuple[str, str], dict[str, str]],
                  fnirs_references: dict[str, dict[str, str]],
                  *, show_plots: bool, save_plots: bool, plot_root: Path, recording=None):
    source = REPOSITORY_ROOT / trial["physio_filepath"]
    if recording is None:
        recording = read_biosignalsplux_hdf5(source)
    start, end = int(trial["physio_start_sample"]), int(trial["physio_end_sample"])
    fs = float(trial["sampling_rate_hz"])
    features: dict[str, list[dict[str, object]]] = {name: [] for name in MODALITIES}
    manifest = []
    common = base_feature(trial, group)
    signal_names = {"ecg": "ecg", "eda": "eda", "resp": "rip", "temp": "temp"}
    for modality in selected_modalities:
        plot_dir = output_directory(plot_root, modality, trial)
        row = {
            **{key: common[key] for key in ("participant_id", "group", "phase", "trial_number", "source_hdf5", "start_sample", "end_sample")},
            "modality": modality, "included": "True", "processing_status": "pending",
            "exclusion_reason": "", "sampling_rate_input": f"{fs:g}",
            "sampling_rate_output": "", "processing_version": "MultimodalPhysioKit 0.1.0",
            "processing_parameters": "", "output_path": plot_dir.relative_to(REPOSITORY_ROOT).as_posix() if save_plots else "",
            "ecg_multiplier_applied": "", "eda_saturation_status": "",
            "eda_processing_decision": "", "fnirs_processing_stage": "",
            "fnirs_reference_method": "", "age_years": "",
            "fnirs_reference_start_sample": "", "fnirs_reference_end_sample": "",
            "age_source": "", "red_reference": "", "infrared_reference": "",
            "dpf_660nm": "", "dpf_860nm": "", "error_message": "",
        }
        try:
            if modality == "eda":
                policy = eda_policy[(trial["participant_id"], trial["phase"])]
                row["eda_saturation_status"] = policy["eda_saturation_status"]
                row["eda_processing_decision"] = policy["eda_processing_decision"]
                if not bool_value(policy["eda_include_for_processing"]):
                    row.update(included="False", processing_status="excluded",
                               exclusion_reason="saturated_eda")
                    manifest.append(row); continue
                if policy["eda_saturation_status"] == "possible_saturation":
                    print(f"QC WARNING: {trial['participant_id']}/{trial['phase']} possible EDA saturation; included by explicit processing policy.")

            if modality == "fnirs":
                reference = fnirs_references[trial["participant_id"]]
                row.update(age_years=reference["age_years"],
                           fnirs_reference_method=reference["reference_method"],
                           fnirs_reference_start_sample=reference["reference_start_sample"],
                           fnirs_reference_end_sample=reference["reference_end_sample"],
                           age_source=reference["age_source"], red_reference=reference["red_reference"],
                           infrared_reference=reference["infrared_reference"],
                           dpf_660nm=reference["dpf_660nm"], dpf_860nm=reference["dpf_860nm"])
                if not reference["fnirs_reference_status"].startswith("valid"):
                    row.update(included="False", processing_status="excluded",
                               exclusion_reason=reference["fnirs_reference_status"],
                               fnirs_processing_stage="not_processed_invalid_participant_reference")
                    manifest.append(row); continue
                red = segment_signal(recording.get_signal("fnirs_red"), start, end)
                infrared = segment_signal(recording.get_signal("fnirs_infrared"), start, end)
                result = process_fnirs(red, infrared, age_years=float(reference["age_years"]),
                                       red_reference=float(reference["red_reference"]),
                                       infrared_reference=float(reference["infrared_reference"]),
                                       show_plots=show_plots, save_plots=save_plots,
                                       output_dir=plot_dir)
                row.update(processing_status="success", sampling_rate_output="10",
                           fnirs_processing_stage="delta_hbo_delta_hbr_features",
                           processing_parameters="provided participant baseline reference; package defaults")
                feature = {**common, "age_years": float(reference["age_years"]),
                           "age_source": reference["age_source"],
                           "fnirs_reference_start_sample": int(reference["reference_start_sample"]),
                           "fnirs_reference_end_sample": int(reference["reference_end_sample"]),
                           "reference_method": reference["reference_method"],
                           "red_reference": float(reference["red_reference"]),
                           "infrared_reference": float(reference["infrared_reference"]),
                           "dpf_660nm": float(reference["dpf_660nm"]),
                           "dpf_860nm": float(reference["dpf_860nm"]),
                           **{key: float(value) for key, value in result.features.items()}}
                features[modality].append(feature); manifest.append(row); continue

            signal = segment_signal(recording.get_signal(signal_names[modality]), start, end)
            if modality == "ecg":
                multiplier = get_ecg_multiplier(trial["participant_id"], ecg_config)
                result = process_ecg(signal, multiplier, show_plots=show_plots,
                                     save_plots=save_plots, output_dir=plot_dir)
                row["ecg_multiplier_applied"] = multiplier
                row["sampling_rate_output"] = f"{fs:g}"
                row["processing_parameters"] = "label_frequency_analysis=2; polarity corrected in memory"
            elif modality == "eda":
                result = process_eda(signal, show_plots=show_plots,
                                     save_plots=save_plots, output_dir=plot_dir)
                row["sampling_rate_output"] = result.metadata["processed_sampling_frequency"]
                row["processing_parameters"] = "cvxEDA=1.1.0; minimum_scr_amplitude=0.01uS"
            elif modality == "resp":
                result = process_resp(signal, show_plots=show_plots,
                                      save_plots=save_plots, output_dir=plot_dir)
                row["sampling_rate_output"] = result.metadata["processed_sampling_frequency"]
                row["processing_parameters"] = (
                    "min_breathing_rate=3.0; max_breathing_rate=30.0; output=100Hz"
                )
            else:
                result = process_temp(signal, show_plots=show_plots,
                                      save_plots=save_plots, output_dir=plot_dir)
                row["sampling_rate_output"] = f"{fs:g}"
                row["processing_parameters"] = "package defaults; 1Hz low-pass"
            feature = {**common, **{key: float(value) for key, value in result.features.items()}}
            features[modality].append(feature)
            row["processing_status"] = "success"
        except Exception as exc:
            row.update(processing_status="failed", error_message=f"{type(exc).__name__}: {exc}")
            print(f"ERROR {trial['participant_id']}/{trial['phase']}/T{trial['trial_number_chronological']}/{modality}: {row['error_message']}")
            if show_plots:
                traceback.print_exc()
        manifest.append(row)
        if show_plots:
            plt.close("all")
    return features, manifest


def baseline_manifest(physio: list[dict[str, str]], groups: dict[str, str]):
    rows=[]
    for source in physio:
        if source["phase"] != "baseline": continue
        for modality in MODALITIES:
            rows.append({"participant_id":source["participant_id"],"group":groups.get(source["participant_id"],""),
                "phase":"baseline","trial_number":"","modality":modality,"included":"False",
                "processing_status":"qc_only_not_feature_extracted","exclusion_reason":"no_baseline_interval_defined",
                "source_hdf5":source["physio_filepath"],"start_sample":0,"end_sample":source["n_samples"],
                "sampling_rate_input":source["sampling_rate_hz"],"sampling_rate_output":"",
                "processing_version":"MultimodalPhysioKit 0.1.0","processing_parameters":"",
                "output_path":"","ecg_multiplier_applied":"","eda_saturation_status":"",
                "eda_processing_decision":"","fnirs_processing_stage":"raw_continuous_qc_only" if modality=="fnirs" else "",
                "fnirs_reference_method":"unresolved_no_justified_reference" if modality=="fnirs" else "",
                "age_years":"","fnirs_reference_start_sample":"","fnirs_reference_end_sample":"",
                "age_source":"","red_reference":"","infrared_reference":"",
                "dpf_660nm":"","dpf_860nm":"",
                "error_message":""})
    return rows


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True); mode.add_argument("--all",action="store_true"); mode.add_argument("--subject",type=int)
    parser.add_argument("--phase",choices=("baseline",)+EXPERIMENTAL_PHASES); parser.add_argument("--trial",type=int)
    parser.add_argument("--modality",choices=MODALITIES+("all",),default="all")
    parser.add_argument("--show-plots",action="store_true"); parser.add_argument("--save-plots",action="store_true")
    parser.add_argument("--overwrite",action="store_true")
    args=parser.parse_args()
    if not args.all and not args.phase: parser.error("--phase is required with --subject")
    if not args.all and args.phase != "baseline" and args.trial is None: parser.error("--trial is required for experimental phases")
    if args.all and args.show_plots: parser.error("--show-plots is not allowed in batch mode")
    describe_api()
    metadata=REPOSITORY_ROOT/"metadata"; plot_root=REPOSITORY_ROOT/"outputs/processing"
    trials=[r for r in read_csv_rows(metadata/"trial_index.csv") if bool_value(r["include_physio_analysis"])]
    physio=read_csv_rows(metadata/"physiological_recordings.csv")
    eda_rows=read_csv_rows(metadata/"eda_saturation_qc.csv"); eda_policy={(r["participant_id"],r["phase"]):r for r in eda_rows}
    reference_rows=read_csv_rows(metadata/"fnirs_baseline_reference.csv"); fnirs_references={r["participant_id"]:r for r in reference_rows}
    participants=participant_metadata_from_xlsx(REPOSITORY_ROOT/"data/SecondRun_Professional.xlsx")
    groups={r["participant_id"]:r["group"] for r in participants}
    ages={r["participant_id"]:(float(r["age_years"]) if r["age_years"] else None) for r in participants}
    modalities=MODALITIES if args.modality=="all" else (args.modality,)
    if args.all:
        targets=trials
        outputs=[metadata/f"features_{m}.csv" for m in modalities]+[metadata/"physiological_processing_manifest.csv"]
        if not args.overwrite and any(p.exists() for p in outputs): raise FileExistsError("Batch outputs exist; use --overwrite")
    elif args.phase=="baseline":
        print("Baseline is continuous QC-only: no interval or features are inferred."); return 0
    else:
        pid=f"Subject_{args.subject}"; targets=[r for r in trials if r["participant_id"]==pid and r["phase"]==args.phase and int(r["trial_number_chronological"])==args.trial]
        if len(targets)!=1: raise ValueError(f"Expected one included trial, found {len(targets)}")
    all_features={m:[] for m in MODALITIES}; manifest=[]
    cached_path = None
    cached_recording = None
    for index,trial in enumerate(targets,1):
        source_path = trial["physio_filepath"]
        if source_path != cached_path:
            cached_recording = read_biosignalsplux_hdf5(REPOSITORY_ROOT / source_path)
            cached_path = source_path
        found,rows=process_trial(trial,modalities,groups.get(trial["participant_id"],""),
                                 ages.get(trial["participant_id"]), metadata/"ecg_polarity_qc.csv",eda_policy,
                                 fnirs_references,
                                 show_plots=args.show_plots,save_plots=args.save_plots,plot_root=plot_root,
                                 recording=cached_recording)
        for modality in MODALITIES: all_features[modality].extend(found[modality])
        manifest.extend(rows)
        if args.all and index%50==0: print(f"Processed {index}/{len(targets)} trials")
    if args.all:
        manifest.extend([r for r in baseline_manifest(physio,groups) if r["modality"] in modalities])
        if modalities != MODALITIES and (metadata/"physiological_processing_manifest.csv").exists():
            prior=read_csv_rows(metadata/"physiological_processing_manifest.csv")
            manifest=[r for r in prior if r["modality"] not in modalities]+manifest
        for modality in modalities: write_dynamic_csv(metadata/f"features_{modality}.csv",all_features[modality],BASE_FIELDS)
        with (metadata/"physiological_processing_manifest.csv").open("w",encoding="utf-8",newline="") as handle:
            writer=csv.DictWriter(handle,fieldnames=MANIFEST_FIELDS); writer.writeheader(); writer.writerows(manifest)
        with (metadata/"participant_metadata.csv").open("w",encoding="utf-8",newline="") as handle:
            writer=csv.DictWriter(handle,fieldnames=("participant_id","group","age_years"));writer.writeheader();writer.writerows(participants)
        counts={m:{s:sum(r["modality"]==m and r["processing_status"]==s for r in manifest) for s in {r["processing_status"] for r in manifest}} for m in MODALITIES}
        log_processing(metadata/"processing_log.csv","PHYSIOLOGICAL_PROCESSING_BATCH",Path(__file__).name,
                       "metadata/physiological_processing_manifest.csv","success",
                       f"valid_trials={len(trials)}; status_counts={json.dumps(counts,sort_keys=True)}; raw_files_modified=false")
        print("Batch status:",json.dumps(counts,indent=2,sort_keys=True))
    else:
        for row in manifest: print(json.dumps(row,indent=2))
        for modality,rows in all_features.items():
            for row in rows: print(modality,"features",json.dumps(row,indent=2))
    return 0


if __name__=="__main__": raise SystemExit(main())
