#!/usr/bin/env python3
"""Save only MultimodalPhysioKit-native processing figures for validated trials."""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import multimodalphysiokit
from multimodalphysiokit.io import read_biosignalsplux_hdf5

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dataset.ecg_polarity import get_ecg_multiplier  # noqa: E402
from dataset.provenance import log_processing  # noqa: E402
from processing.common import participant_metadata_from_xlsx, read_csv_rows, segment_signal  # noqa: E402
from processing.ecg import process_ecg  # noqa: E402
from processing.eda import process_eda  # noqa: E402
from processing.fnirs import process_fnirs  # noqa: E402
from processing.resp import process_resp  # noqa: E402
from processing.temp import process_temp  # noqa: E402

MODALITIES = ("ecg", "eda", "resp", "temp", "fnirs")
FIELDS = (
    "participant_id", "group", "phase", "trial_number", "modality",
    "package_processor", "package_method", "processing_status", "plot_index",
    "plot_path", "plot_origin", "source_hdf5", "start_sample", "end_sample",
    "package_version", "exception_type", "exception_message",
)


def truth(value: str) -> bool:
    return value.strip().lower() == "true"


def output_dir(trial: dict[str, str], modality: str) -> Path:
    return (ROOT / "outputs/multimodalphysiokit_debug" / trial["participant_id"] /
            trial["phase"] / f"trial_{int(trial['trial_number_chronological']):02d}" / modality)


def processor_name(modality: str) -> str:
    return {"ecg":"ECGProcessor", "eda":"EDAProcessor", "resp":"RespirationProcessor",
            "temp":"TemperatureProcessor", "fnirs":"FNIRSProcessor"}[modality]


def run_modality(trial: dict[str, str], modality: str, recording, *, group: str,
                 show_plots: bool, save_plots: bool, overwrite: bool,
                 ecg_config: Path, eda_policy, references):
    destination=output_dir(trial,modality)
    existing=sorted(destination.glob("*.png")) if destination.exists() else []
    if existing and not overwrite:
        return "skipped_existing", "", "", existing
    if overwrite and destination.exists(): shutil.rmtree(destination)
    if save_plots: destination.mkdir(parents=True,exist_ok=True)
    start,end=int(trial["physio_start_sample"]),int(trial["physio_end_sample"])
    before=set(plt.get_fignums()); status="success"; etype="";emessage=""
    try:
        if modality=="eda":
            policy=eda_policy[(trial["participant_id"],trial["phase"])]
            if not truth(policy["eda_include_for_processing"]):
                return "excluded", "", "saturated_eda", []
        if modality=="fnirs":
            reference=references[trial["participant_id"]]
            if not reference["fnirs_reference_status"].startswith("valid"):
                return "excluded", "", reference["fnirs_reference_status"], []
            red=segment_signal(recording.get_signal("fnirs_red"),start,end)
            infrared=segment_signal(recording.get_signal("fnirs_infrared"),start,end)
            process_fnirs(red,infrared,age_years=float(reference["age_years"]),
                          red_reference=float(reference["red_reference"]),
                          infrared_reference=float(reference["infrared_reference"]),
                          show_plots=show_plots,save_plots=save_plots,output_dir=destination)
        else:
            label={"ecg":"ecg","eda":"eda","resp":"rip","temp":"temp"}[modality]
            signal=segment_signal(recording.get_signal(label),start,end)
            if modality=="ecg":
                process_ecg(signal,get_ecg_multiplier(trial["participant_id"],ecg_config),
                            show_plots=show_plots,save_plots=save_plots,output_dir=destination)
            elif modality=="eda": process_eda(signal,show_plots=show_plots,save_plots=save_plots,output_dir=destination)
            elif modality=="resp": process_resp(signal,show_plots=show_plots,save_plots=save_plots,output_dir=destination)
            else: process_temp(signal,show_plots=show_plots,save_plots=save_plots,output_dir=destination)
    except Exception as exc:
        status="failed";etype=type(exc).__name__;emessage=str(exc)
    finally:
        # Figures are package-created. Closing them changes neither content nor processing.
        for number in set(plt.get_fignums())-before: plt.close(number)
    files=sorted(destination.glob("*.png")) if destination.exists() else []
    return status,etype,emessage,files


def main()->int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all",action="store_true");parser.add_argument("--subject",type=int)
    parser.add_argument("--phase",choices=("pre_test","test_1","test_2","test_3","evaluation"));parser.add_argument("--trial",type=int)
    parser.add_argument("--modality",choices=MODALITIES+("all",),default="all")
    parser.add_argument("--show-plots",action="store_true");parser.add_argument("--save-plots",action="store_true");parser.add_argument("--overwrite",action="store_true")
    args=parser.parse_args()
    if not args.all and args.subject is None: parser.error("use --all or --subject")
    if args.trial is not None and args.phase is None: parser.error("--trial requires --phase")
    if args.all and args.show_plots: parser.error("--show-plots is not allowed with --all")
    if args.all and not args.save_plots: parser.error("full batch requires --save-plots")
    metadata=ROOT/"metadata"
    trials=[r for r in read_csv_rows(metadata/"trial_index.csv") if truth(r["include_physio_analysis"])]
    if args.subject is not None: trials=[r for r in trials if r["participant_id"]==f"Subject_{args.subject}"]
    if args.phase: trials=[r for r in trials if r["phase"]==args.phase]
    if args.trial is not None: trials=[r for r in trials if int(r["trial_number_chronological"])==args.trial]
    if not trials: raise ValueError("No eligible trials match selection")
    participants=participant_metadata_from_xlsx(ROOT/"data/SecondRun_Professional.xlsx")
    groups={r["participant_id"]:r["group"] for r in participants}
    policies={(r["participant_id"],r["phase"]):r for r in read_csv_rows(metadata/"eda_saturation_qc.csv")}
    references={r["participant_id"]:r for r in read_csv_rows(metadata/"fnirs_baseline_reference.csv")}
    modalities=MODALITIES if args.modality=="all" else (args.modality,)
    rows=[]; failures=[]; cached_path=None;recording=None
    for index,trial in enumerate(trials,1):
        if trial["physio_filepath"]!=cached_path:
            recording=read_biosignalsplux_hdf5(ROOT/trial["physio_filepath"]);cached_path=trial["physio_filepath"]
        for modality in modalities:
            status,etype,message,files=run_modality(trial,modality,recording,group=groups[trial["participant_id"]],
                show_plots=args.show_plots,save_plots=args.save_plots,overwrite=args.overwrite,
                ecg_config=metadata/"ecg_polarity_qc.csv",eda_policy=policies,references=references)
            if status in {"failed","excluded"}: failures.append((trial["participant_id"],trial["phase"],trial["trial_number_chronological"],modality,status,etype,message))
            for plot_index,path in enumerate(files,1):
                rows.append({"participant_id":trial["participant_id"],"group":groups[trial["participant_id"]],"phase":trial["phase"],
                  "trial_number":trial["trial_number_chronological"],"modality":modality,"package_processor":processor_name(modality),
                  "package_method":"process","processing_status":status,"plot_index":plot_index,"plot_path":path.relative_to(ROOT).as_posix(),
                  "plot_origin":"MultimodalPhysioKit","source_hdf5":trial["physio_filepath"],"start_sample":trial["physio_start_sample"],
                  "end_sample":trial["physio_end_sample"],"package_version":multimodalphysiokit.__version__,"exception_type":etype,"exception_message":message})
        if args.save_plots and index%25==0: print(f"Processed {index}/{len(trials)} trials; native plots={len(rows)}")
    if args.all:
        path=metadata/"multimodalphysiokit_debug_plot_manifest.csv"
        with path.open("w",encoding="utf-8",newline="") as f:w=csv.DictWriter(f,fieldnames=FIELDS);w.writeheader();w.writerows(rows)
        log_processing(metadata/"processing_log.csv", "MULTIMODALPHYSIOKIT_NATIVE_DEBUG_PLOTS",
                       Path(__file__).name, "metadata/multimodalphysiokit_debug_plot_manifest.csv",
                       "success", f"eligible_trials={len(trials)}; native_plots={len(rows)}; "
                       f"failed_or_excluded_calls={len(failures)}; plot_origin=MultimodalPhysioKit; raw_files_modified=false")
    print(f"Eligible trials inspected: {len(trials)}; package-native plots: {len(rows)}; failed/excluded calls: {len(failures)}")
    for failure in failures: print("FAILURE",*failure)
    return 0

if __name__=="__main__":raise SystemExit(main())
