#!/usr/bin/env python3
"""Extract authorized baseline features and assemble non-destructive final tables."""

from __future__ import annotations

import csv
import shutil
import sys
from collections import Counter
from contextlib import redirect_stderr,redirect_stdout
from io import StringIO
from pathlib import Path

import matplotlib.pyplot as plt
import multimodalphysiokit
import numpy as np
from multimodalphysiokit.core import Signal
from multimodalphysiokit.io import read_biosignalsplux_hdf5
from multimodalphysiokit.processors import ECGProcessor,EDAProcessor,FNIRSProcessor,RespirationProcessor,TemperatureProcessor
from multimodalphysiokit.processors.fnirs import _differential_pathlength_factor

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from dataset.baseline_events import read_baseline_rising_edges  # noqa:E402
from dataset.ecg_polarity import get_ecg_multiplier  # noqa:E402
from dataset.provenance import log_processing  # noqa:E402
from processing.common import participant_metadata_from_xlsx,read_csv_rows  # noqa:E402

MODALITIES=("ecg","eda","resp","temp","fnirs");ID_FIELDS=("participant_id","group","phase","trial_number")

def truth(v):return str(v).lower()=="true"
def write_rows(path,rows,preferred=()):
    fields=list(preferred)+sorted({k for r in rows for k in r if k not in preferred})
    with path.open("w",newline="",encoding="utf-8") as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def corrected_recording(path,pid,config):
    rec=read_biosignalsplux_hdf5(path,name=f"{pid}_baseline");ecg=rec.get_signal("ecg");m=get_ecg_multiplier(pid,config)
    rec.add_signal("ecg",Signal(samples=ecg.samples*m,time=ecg.time.copy(),sampling_frequency=ecg.sampling_frequency,label="ecg_corrected",units=ecg.units,metadata={**ecg.metadata,"ecg_multiplier_final":m,"polarity_correction":"in_memory"}));return rec
def processor(modality,age,reference=None):
    if modality=="ecg":return ECGProcessor(label_frequency_analysis=2,return_intermediates=True),("ecg",),{}
    if modality=="eda":return EDAProcessor(minimum_scr_amplitude=.01),("eda",),{}
    if modality=="resp":return RespirationProcessor(min_breathing_rate=3.0,max_breathing_rate=30.0),("rip",),{}
    if modality=="temp":return TemperatureProcessor(),("temp",),{}
    return FNIRSProcessor(age_years=age),("fnirs_red","fnirs_infrared"),{"baseline_red_value":reference[0],"baseline_infrared_value":reference[1]}
def move_plots(stage,destination,prefix):
    paths=[]
    for source in sorted(stage.glob("**/*.png")):
        target=destination/f"{prefix}_{source.name}";destination.mkdir(parents=True,exist_ok=True);source.replace(target);paths.append(target)
    if stage.exists():shutil.rmtree(stage)
    return paths
def process_phase(rec,modality,age,reference,stage,save=True):
    proc,names,kwargs=processor(modality,age,reference);kwargs.update(show_plots=False,save_plots=save,output_dir=stage)
    if modality in {"ecg","eda","fnirs"}:kwargs["output_label"]=modality
    before=set(plt.get_fignums())
    try:
        if modality=="eda":
            with redirect_stdout(StringIO()),redirect_stderr(StringIO()):return rec.process_phases(proc,names,phases=["baseline"],process_kwargs=kwargs)["baseline"]
        return rec.process_phases(proc,names,phases=["baseline"],process_kwargs=kwargs)["baseline"]
    finally:
        for n in set(plt.get_fignums())-before:plt.close(n)

def main():
    out=ROOT/"outputs/features";metadata=ROOT/"metadata"
    if out.exists():raise FileExistsError("outputs/features already exists; refusing to overwrite")
    for m in MODALITIES:(out/m).mkdir(parents=True)
    plot_root=out/"baseline_processing_plots"
    for m in MODALITIES:(plot_root/m).mkdir(parents=True)
    people=participant_metadata_from_xlsx(ROOT/"data/SecondRun_Professional.xlsx");groups={r["participant_id"]:r["group"] for r in people};ages={r["participant_id"]:float(r["age_years"]) for r in people}
    phys={(r["participant_id"],r["phase"]):r for r in read_csv_rows(metadata/"physiological_recordings.csv")};eda_qc={(r["participant_id"],r["phase"]):r for r in read_csv_rows(metadata/"eda_saturation_qc.csv")}
    intervals=[];references=[];baseline={m:[] for m in MODALITIES}
    for person in people:
        pid=person["participant_id"];source=phys[(pid,"baseline")];path=ROOT/source["physio_filepath"]
        events,fs,n=read_baseline_rising_edges(path);first=events[0];start=first.event_sample+round(30*fs);target=first.event_sample+round(210*fs);end=min(target,n);duration=(end-start)/fs
        status="valid_full" if end==target else ("valid_shortened" if duration>120 else "excluded_insufficient_baseline")
        intervals.append({"participant_id":pid,"first_marker_time_s":first.event_time_s,"baseline_start_s":start/fs,"baseline_end_s":end/fs,"baseline_duration_s":duration,"interval_status":status})
        if not status.startswith("valid"):
            for m in MODALITIES:baseline[m].append({"participant_id":pid,"group":groups[pid],"phase":"baseline","trial_number":"","baseline_start_s":start/fs,"baseline_end_s":end/fs,"baseline_duration_s":duration,"processing_status":"excluded","processing_reason":status,"source_hdf5":source["physio_filepath"]})
            references.append({"participant_id":pid,"age":ages[pid],"reference_start_s":start/fs,"reference_end_s":end/fs,"reference_duration_s":duration,"reference_status":status,"I0_red":"","I0_infrared":""});continue
        # Recording phases are half-open.  Express the authorized [start, end)
        # sample interval with an endpoint infinitesimally below end/fs.  This
        # still maps to stop index ``end`` while avoiding floating-point cases
        # where an end-of-recording endpoint compares greater than signal_stop.
        phase_end_s=np.nextafter(end/fs,-np.inf)
        rec=corrected_recording(path,pid,metadata/"ecg_polarity_qc.csv");rec.set_phases(np.asarray([[start/fs],[phase_end_s]]),["baseline"]);red=rec.extract_phase("fnirs_red","baseline");infrared=rec.extract_phase("fnirs_infrared","baseline")
        fnirs_proc=FNIRSProcessor(age_years=ages[pid]);i0_red,i0_infrared=fnirs_proc.compute_baseline_values(red,infrared);reference=(i0_red,i0_infrared)
        references.append({"participant_id":pid,"age":ages[pid],"reference_start_s":start/fs,"reference_end_s":end/fs,"reference_duration_s":duration,"reference_status":status,"I0_red":i0_red,"I0_infrared":i0_infrared})
        for m in MODALITIES:
            base={"participant_id":pid,"group":groups[pid],"phase":"baseline","trial_number":"","baseline_start_s":start/fs,"baseline_end_s":end/fs,"baseline_duration_s":duration,"processing_status":"pending","processing_reason":"","source_hdf5":source["physio_filepath"]}
            if m=="eda" and not truth(eda_qc[(pid,"baseline")]["eda_include_for_processing"]):base.update(processing_status="excluded",processing_reason="saturated_eda");baseline[m].append(base);continue
            stage=plot_root/".stage"/m/pid;result=None
            try:
                result=process_phase(rec,m,ages[pid],reference,stage);base.update(processing_status="success",**{k:float(v) for k,v in result.features.items()})
            except Exception as exc:base.update(processing_status="failed",processing_reason=f"{type(exc).__name__}: {exc}")
            paths=move_plots(stage,plot_root/m,f"{pid}_baseline");base["native_plot_paths"]=";".join(p.relative_to(ROOT).as_posix() for p in paths);baseline[m].append(base)
    write_rows(out/"baseline_interval_manifest.csv",intervals,("participant_id","first_marker_time_s","baseline_start_s","baseline_end_s","baseline_duration_s","interval_status"))
    write_rows(out/"fnirs/fnirs_baseline_reference_final.csv",references,("participant_id","age","reference_start_s","reference_end_s","reference_duration_s","reference_status","I0_red","I0_infrared"))
    refs={r["participant_id"]:r for r in references};trials=[r for r in read_csv_rows(metadata/"trial_index.csv") if truth(r["include_physio_analysis"])]
    # Copy four authoritative, numerically validated production feature tables unchanged.
    for m in ("ecg","eda","resp","temp"):shutil.copy2(metadata/f"features_{m}.csv",out/m/"experimental_features.csv")
    # Independently regenerate experimental fNIRS with the final baseline I0 pair supplied.
    fnirs_experimental=[]
    for source in (r for r in phys.values() if r["phase"]!="baseline"):
        pid=source["participant_id"];phase=source["phase"];selected=sorted([r for r in trials if r["participant_id"]==pid and r["phase"]==phase],key=lambda r:int(r["trial_number_chronological"]))
        rec=read_biosignalsplux_hdf5(ROOT/source["physio_filepath"],name=f"{pid}_{phase}");f=rec.get_signal("fnirs_red").sampling_frequency
        rec.set_phases(np.asarray([[int(r["physio_start_sample"])/f for r in selected],[int(r["physio_end_sample"])/f for r in selected]]),[f"trial_{int(r['trial_number_chronological']):02d}" for r in selected])
        ref=refs[pid];proc=FNIRSProcessor(age_years=ages[pid]);results=rec.process_phases(proc,("fnirs_red","fnirs_infrared"),process_kwargs={"baseline_red_value":float(ref["I0_red"]),"baseline_infrared_value":float(ref["I0_infrared"]),"show_plots":False,"save_plots":False})
        for trial in selected:
            label=f"trial_{int(trial['trial_number_chronological']):02d}";result=results[label];start=int(trial["physio_start_sample"]);end=int(trial["physio_end_sample"])
            fnirs_experimental.append({"participant_id":pid,"group":groups[pid],"phase":phase,"trial_number":trial["trial_number_chronological"],"source_hdf5":source["physio_filepath"],"start_sample":start,"end_sample":end,"duration_s":(end-start)/f,"age_years":ages[pid],"age_source":"data/SecondRun_Professional.xlsx:Age","fnirs_reference_start_sample":round(float(ref["reference_start_s"])*f),"fnirs_reference_end_sample":round(float(ref["reference_end_s"])*f),"reference_method":"FNIRSProcessor.compute_baseline_values","red_reference":float(ref["I0_red"]),"infrared_reference":float(ref["I0_infrared"]),"dpf_660nm":_differential_pathlength_factor(660,ages[pid]),"dpf_860nm":_differential_pathlength_factor(860,ages[pid]),**{k:float(v) for k,v in result.features.items()}})
    write_rows(out/"fnirs/experimental_features.csv",fnirs_experimental,("participant_id","group","phase","trial_number","source_hdf5","start_sample","end_sample","duration_s"))
    for m in MODALITIES:
        write_rows(out/m/"baseline_features.csv",baseline[m],("participant_id","group","phase","trial_number","baseline_start_s","baseline_end_s","baseline_duration_s","processing_status","processing_reason","source_hdf5"))
        experimental=read_csv_rows(out/m/"experimental_features.csv");write_rows(out/m/"features_all.csv",baseline[m]+experimental,ID_FIELDS)
    inventory=[]
    units={"ecg":"mixed ECG/HRV units","eda":"mixed EDA/SCR units","resp":"mixed respiration units","temp":"degC and derivatives","fnirs":"package concentration-change feature units"}
    for m in MODALITIES:
        exp=read_csv_rows(out/m/"experimental_features.csv");b=baseline[m];exp_features=set(exp[0])-set(("participant_id","group","phase","trial_number","source_hdf5","start_sample","end_sample","duration_s","age_years","age_source","fnirs_reference_start_sample","fnirs_reference_end_sample","reference_method","red_reference","infrared_reference","dpf_660nm","dpf_860nm"));b_features={k for r in b if r["processing_status"]=="success" for k in r}-set(("participant_id","group","phase","trial_number","baseline_start_s","baseline_end_s","baseline_duration_s","processing_status","processing_reason","source_hdf5","native_plot_paths"))
        for feature in sorted(exp_features|b_features):inventory.append({"modality":m,"feature_name":feature,"source_processor":{"ecg":"ECGProcessor(mode=2)","eda":"EDAProcessor","resp":"RespirationProcessor","temp":"TemperatureProcessor","fnirs":"FNIRSProcessor"}[m],"baseline_available":feature in b_features,"experimental_available":feature in exp_features,"units_if_known":units[m],"notes":"native feature"})
    write_rows(out/"feature_inventory.csv",inventory,("modality","feature_name","source_processor","baseline_available","experimental_available","units_if_known","notes"))
    manifest=read_csv_rows(metadata/"physiological_processing_manifest.csv");complete=[]
    for person in people:
        pid=person["participant_id"]
        for m in MODALITIES:
            mx=[r for r in manifest if r["participant_id"]==pid and r["modality"]==m and r["phase"]!="baseline"]
            br=next(r for r in baseline[m] if r["participant_id"]==pid)
            complete.append({"participant_id":pid,"modality":m,"baseline_expected":True,"baseline_successful":br["processing_status"]=="success","experimental_trials_expected":len(mx),"experimental_trials_successful":sum(r["processing_status"]=="success" for r in mx),"experimental_trials_excluded":sum(r["processing_status"]=="excluded" for r in mx),"experimental_trials_failed":sum(r["processing_status"]=="failed" for r in mx)})
    write_rows(out/"feature_completeness.csv",complete,("participant_id","modality","baseline_expected","baseline_successful","experimental_trials_expected","experimental_trials_successful","experimental_trials_excluded","experimental_trials_failed"))
    stage=plot_root/".stage"
    if stage.exists():
        for d in sorted((p for p in stage.glob("**/*") if p.is_dir()),key=lambda p:len(p.parts),reverse=True):d.rmdir()
        stage.rmdir()
    counts={m:Counter(r["processing_status"] for r in baseline[m]) for m in MODALITIES};log_processing(metadata/"processing_log.csv","BASELINE_FEATURE_EXTRACTION",Path(__file__).name,"outputs/features","success",f"intervals={Counter(r['interval_status'] for r in intervals)}; baseline_status={counts}; experimental_fnirs={len(fnirs_experimental)}; raw_files_modified=false")
    print("intervals",Counter(r["interval_status"] for r in intervals));print("baseline",counts);print("experimental_fnirs",len(fnirs_experimental));return 0
if __name__=="__main__":raise SystemExit(main())
