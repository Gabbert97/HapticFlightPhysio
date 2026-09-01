#!/usr/bin/env python3
"""Validate and process behavioral phases through MultimodalPhysioKit Recording."""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path

import matplotlib.pyplot as plt
import multimodalphysiokit
import numpy as np
from multimodalphysiokit.core import Signal
from multimodalphysiokit.io import read_biosignalsplux_hdf5
from multimodalphysiokit.processors import (ECGProcessor, EDAProcessor, FNIRSProcessor,
                                            RespirationProcessor, TemperatureProcessor)
from multimodalphysiokit.utils.plotting import plot_signal_with_phases

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from dataset.ecg_polarity import get_ecg_multiplier  # noqa:E402
from dataset.provenance import log_processing  # noqa:E402
from processing.common import participant_metadata_from_xlsx,read_csv_rows  # noqa:E402

MODALITIES=("ecg","eda","resp","temp","fnirs")
PHASES=("baseline","pre_test","test_1","test_2","test_3","evaluation")
FEATURE_ID=("participant_id","group","phase","trial_number","source_hdf5","start_sample","end_sample","duration_s")
PROCESS_FIELDS=("participant_id","group","phase","trial_number","modality","source_hdf5","recording_phase_start_s","recording_phase_end_s","recording_start_sample","recording_end_sample","processing_status","processing_reason","n_features","feature_output_file","n_native_plots","native_plot_paths","package_version","processing_origin","exception_type","exception_message")

def truth(v):return str(v).lower()=="true"
def write_dynamic(path,rows,base):
    fields=list(base)+sorted({k for r in rows for k in r if k not in base})
    with path.open("w",newline="",encoding="utf-8") as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def corrected_recording(source:Path,pid:str,ecg_config:Path):
    rec=read_biosignalsplux_hdf5(source,name=pid)
    ecg=rec.get_signal("ecg");mult=get_ecg_multiplier(pid,ecg_config)
    rec.add_signal("ecg",Signal(samples=ecg.samples*mult,time=ecg.time.copy(),sampling_frequency=ecg.sampling_frequency,label=ecg.label,units=ecg.units,metadata={**ecg.metadata,"ecg_multiplier_final":mult,"polarity_correction":"in_memory"}))
    return rec

def configure_phases(rec,trials):
    if not trials:return
    fs=rec.get_signal("ecg").sampling_frequency
    intervals=np.asarray([[int(r["physio_start_sample"])/fs for r in trials],[int(r["physio_end_sample"])/fs for r in trials]])
    rec.set_phases(intervals,[f"trial_{int(r['trial_number_chronological']):02d}" for r in trials])

def move_native(stage:Path,destination:Path,prefix:str,overwrite:bool):
    files=[]
    for source in sorted(stage.glob("**/*.png")):
        target=destination/f"{prefix}_{source.name}"
        if target.exists() and not overwrite:raise FileExistsError(target)
        destination.mkdir(parents=True,exist_ok=True);source.replace(target);files.append(target)
    if stage.exists():shutil.rmtree(stage)
    return files

def recording_plots(rec,pid,phase,output_root,show,save,overwrite):
    if rec.n_phases==0:return [],"native_plot_requires_phases"
    names=("ecg","eda","rip","temp","fnirs_red","fnirs_infrared");paths=[]
    for name in names:
        destination=output_root/phase/f"Subject_{int(pid.split('_')[1]):02d}_{phase}_{name}.png"
        if destination.exists() and not overwrite:raise FileExistsError(destination)
        stage=output_root/".stage"/pid/phase/name
        before=set(plt.get_fignums())
        plot_signal_with_phases(rec,name,show=show,save_plots=save,output_dir=stage)
        for number in set(plt.get_fignums())-before:plt.close(number)
        if save:
            source=stage/f"{name}_phases.png";destination.parent.mkdir(parents=True,exist_ok=True);source.replace(destination);paths.append(destination)
            shutil.rmtree(stage)
    return paths,""

def processor_config(modality,pid,refs):
    if modality=="ecg":return ECGProcessor(label_frequency_analysis=2,return_intermediates=True),("ecg",),{}
    if modality=="eda":return EDAProcessor(minimum_scr_amplitude=.01),("eda",),{}
    if modality=="resp":return RespirationProcessor(min_breathing_rate=3.0,max_breathing_rate=30.0),("rip",),{}
    if modality=="temp":return TemperatureProcessor(),("temp",),{}
    ref=refs[pid]
    return FNIRSProcessor(age_years=float(ref["age_years"])),("fnirs_red","fnirs_infrared"),{"baseline_red_value":float(ref["red_reference"]),"baseline_infrared_value":float(ref["infrared_reference"])}

def compare_features(modality,new_rows,old_path,output_path):
    old=read_csv_rows(old_path);keys=lambda r:(r["participant_id"],r["phase"],str(int(r["trial_number"])))
    om={keys(r):r for r in old};rows=[]
    for new in new_rows:
        prior=om.get(keys(new));
        if not prior:continue
        for feature in sorted(set(new)&set(prior)-set(FEATURE_ID)):
            try:a=float(prior[feature]);b=float(new[feature])
            except (ValueError,TypeError):continue
            absolute=abs(a-b);relative=absolute/max(abs(a),np.finfo(float).eps)
            rows.append({"participant_id":new["participant_id"],"phase":new["phase"],"trial_number":new["trial_number"],"feature":feature,"old_value":a,"recording_value":b,"absolute_difference":absolute,"relative_difference":relative,"equal_or_close":bool(np.isclose(a,b,rtol=1e-7,atol=1e-10))})
    fields=("participant_id","phase","trial_number","feature","old_value","recording_value","absolute_difference","relative_difference","equal_or_close")
    with output_path.open("w",newline="",encoding="utf-8") as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    return rows

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--all",action="store_true");p.add_argument("--subject",type=int);p.add_argument("--phase",choices=PHASES);p.add_argument("--trial",type=int);p.add_argument("--modality",choices=MODALITIES+("all",),default="all");p.add_argument("--show-plots",action="store_true");p.add_argument("--save-plots",action="store_true");p.add_argument("--overwrite",action="store_true");args=p.parse_args()
    if not args.all and args.subject is None:p.error("use --all or --subject")
    if args.trial is not None and args.phase is None:p.error("--trial requires --phase")
    if args.all and args.show_plots:p.error("--show-plots is not allowed with --all")
    metadata=ROOT/"metadata";rec_plot_root=ROOT/"outputs/multimodalphysiokit_recordings";proc_plot_root=ROOT/"outputs/multimodalphysiokit_recording_processing"
    new_files=[metadata/f"recording_features_{m}.csv" for m in MODALITIES]+[metadata/"multimodalphysiokit_recording_manifest.csv",metadata/"multimodalphysiokit_recording_validation.csv",metadata/"multimodalphysiokit_recording_processing_manifest.csv",metadata/"recording_ecg_frequency_validation.csv",metadata/"recording_vs_trial_features_summary.csv"]+[metadata/f"recording_vs_trial_features_{m}.csv" for m in MODALITIES]
    if args.all and not args.overwrite and any(x.exists() for x in new_files):raise FileExistsError("Recording outputs exist; use --overwrite")
    if args.all and args.overwrite:
        for root in (rec_plot_root,proc_plot_root):
            if root.exists():shutil.rmtree(root)
    for phase in PHASES:(rec_plot_root/phase).mkdir(parents=True,exist_ok=True)
    for m in MODALITIES:(proc_plot_root/m).mkdir(parents=True,exist_ok=True)
    participants=participant_metadata_from_xlsx(ROOT/"data/SecondRun_Professional.xlsx");groups={r["participant_id"]:r["group"] for r in participants}
    physio=read_csv_rows(metadata/"physiological_recordings.csv");trials=[r for r in read_csv_rows(metadata/"trial_index.csv") if truth(r["include_physio_analysis"])]
    eda={(r["participant_id"],r["phase"]):r for r in read_csv_rows(metadata/"eda_saturation_qc.csv")};refs={r["participant_id"]:r for r in read_csv_rows(metadata/"fnirs_baseline_reference.csv")}
    if args.subject is not None:physio=[r for r in physio if r["participant_id"]==f"Subject_{args.subject}"]
    if args.phase:physio=[r for r in physio if r["phase"]==args.phase]
    modalities=MODALITIES if args.modality=="all" else (args.modality,)
    validation=[];recording_manifest=[];processing=[];features={m:[] for m in MODALITIES};frequency=[]
    for ri,source in enumerate(physio,1):
        pid=source["participant_id"];phase=source["phase"];phase_trials=sorted([r for r in trials if r["participant_id"]==pid and r["phase"]==phase],key=lambda r:int(r["trial_number_chronological"]))
        if args.trial is not None:phase_trials=[r for r in phase_trials if int(r["trial_number_chronological"])==args.trial]
        rec=corrected_recording(ROOT/source["physio_filepath"],pid,metadata/"ecg_polarity_qc.csv");configure_phases(rec,phase_trials)
        plot_paths,plot_reason=recording_plots(rec,pid,phase,rec_plot_root,args.show_plots,args.save_plots,args.overwrite)
        recording_manifest.append({"participant_id":pid,"group":groups[pid],"recording_name":phase,"source_hdf5":source["physio_filepath"],"n_signals":len(rec),"signal_names":";".join(rec.signal_names()),"n_phases":rec.n_phases,"native_plot_paths":";".join(p.relative_to(ROOT).as_posix() for p in plot_paths),"visualization_status":"success" if plot_paths else (plot_reason or "not_saved")})
        if phase=="baseline":continue
        indexes=rec.phase_indices("ecg") if rec.n_phases else np.empty((2,0),int);fs=rec.get_signal("ecg").sampling_frequency
        for i,t in enumerate(phase_trials):
            expected_start=int(t["physio_start_sample"]);expected_end=int(t["physio_end_sample"]);actual_start=int(indexes[0,i]);actual_end=int(indexes[1,i])
            validation.append({"participant_id":pid,"group":groups[pid],"recording_name":phase,"trial_number":t["trial_number_chronological"],"expected_start_sample":expected_start,"expected_end_sample":expected_end,"recording_start_sample":actual_start,"recording_end_sample":actual_end,"start_difference_samples":actual_start-expected_start,"end_difference_samples":actual_end-expected_end,"duration_s":(actual_end-actual_start)/fs,"validation_status":"valid" if (actual_start,actual_end)==(expected_start,expected_end) else "mismatch"})
        for modality in modalities:
            for t in phase_trials:
                label=f"trial_{int(t['trial_number_chronological']):02d}";idx=rec.phase_labels.index(label);start,end=indexes[:,idx];prefix=f"Subject_{int(pid.split('_')[1]):02d}_{phase}_{label}"
                base={"participant_id":pid,"group":groups[pid],"phase":phase,"trial_number":t["trial_number_chronological"],"modality":modality,"source_hdf5":source["physio_filepath"],"recording_phase_start_s":rec.phase_intervals_seconds[0,idx],"recording_phase_end_s":rec.phase_intervals_seconds[1,idx],"recording_start_sample":int(start),"recording_end_sample":int(end),"processing_status":"pending","processing_reason":"","n_features":0,"feature_output_file":f"metadata/recording_features_{modality}.csv","n_native_plots":0,"native_plot_paths":"","package_version":multimodalphysiokit.__version__,"processing_origin":"Recording.process_phases","exception_type":"","exception_message":""}
                if modality=="eda" and not truth(eda[(pid,phase)]["eda_include_for_processing"]):base.update(processing_status="excluded",processing_reason="saturated_eda");processing.append(base);continue
                processor,names,kwargs=processor_config(modality,pid,refs);stage=proc_plot_root/".stage"/modality/pid/phase/label
                kwargs.update(show_plots=args.show_plots,save_plots=args.save_plots,output_dir=stage,output_label=modality) if modality in {"ecg","eda","fnirs"} else kwargs.update(show_plots=args.show_plots,save_plots=args.save_plots,output_dir=stage)
                before=set(plt.get_fignums());result=None
                try:
                    if modality=="eda" and not args.show_plots:
                        with redirect_stdout(StringIO()),redirect_stderr(StringIO()):result=rec.process_phases(processor,names,phases=[label],process_kwargs=kwargs)[label]
                    else:result=rec.process_phases(processor,names,phases=[label],process_kwargs=kwargs)[label]
                    feat={"participant_id":pid,"group":groups[pid],"phase":phase,"trial_number":t["trial_number_chronological"],"source_hdf5":source["physio_filepath"],"start_sample":int(start),"end_sample":int(end),"duration_s":(end-start)/fs,**{k:float(v) for k,v in result.features.items()}};features[modality].append(feat);base.update(processing_status="success",n_features=len(result.features))
                except Exception as exc:base.update(processing_status="failed",processing_reason="package_exception",exception_type=type(exc).__name__,exception_message=str(exc))
                finally:
                    for number in set(plt.get_fignums())-before:plt.close(number)
                native=move_native(stage,proc_plot_root/modality,prefix,args.overwrite) if args.save_plots else []
                base.update(n_native_plots=len(native),native_plot_paths=";".join(p.relative_to(ROOT).as_posix() for p in native));processing.append(base)
                if modality=="ecg":
                    frow={"participant_id":pid,"phase":phase,"trial_number":t["trial_number_chronological"],"trial_duration_s":(end-start)/fs,"frequency_analysis_status":"success" if result else "failed","failure_reason":"" if result else base["exception_message"],"spectrogram_generated":any("STFT" in p.name for p in native),"spectrogram_path":";".join(p.relative_to(ROOT).as_posix() for p in native if "STFT" in p.name)}
                    if result:
                        frow.update(n_rpeaks=len(result.intermediates["accepted_r_peak_indices"]),n_rr_intervals=len(result.intermediates["ibi"]));frow.update({k:result.features[k] for k in result.features if k.startswith(("ecg_plf","ecg_phf","ecg_lf_hf"))})
                    frequency.append(frow)
        if ri%10==0:print(f"Recordings {ri}/{len(physio)}; manifest rows={len(processing)}")
    if args.all:
        if any(r["validation_status"]!="valid" for r in validation):raise RuntimeError("Phase-index mismatch; feature outputs not finalized")
        write_dynamic(metadata/"multimodalphysiokit_recording_manifest.csv",recording_manifest,("participant_id","group","recording_name","source_hdf5","n_signals","signal_names","n_phases","native_plot_paths","visualization_status"))
        write_dynamic(metadata/"multimodalphysiokit_recording_validation.csv",validation,("participant_id","group","recording_name","trial_number","expected_start_sample","expected_end_sample","recording_start_sample","recording_end_sample","start_difference_samples","end_difference_samples","duration_s","validation_status"))
        with (metadata/"multimodalphysiokit_recording_processing_manifest.csv").open("w",newline="",encoding="utf-8") as f:w=csv.DictWriter(f,fieldnames=PROCESS_FIELDS);w.writeheader();w.writerows(processing)
        comparisons=[]
        for m in MODALITIES:
            write_dynamic(metadata/f"recording_features_{m}.csv",features[m],FEATURE_ID)
            comparisons+= [{**r,"modality":m} for r in compare_features(m,features[m],metadata/f"features_{m}.csv",metadata/f"recording_vs_trial_features_{m}.csv")]
        write_dynamic(metadata/"recording_ecg_frequency_validation.csv",frequency,("participant_id","phase","trial_number","trial_duration_s","n_rpeaks","n_rr_intervals","frequency_analysis_status","failure_reason","spectrogram_generated","spectrogram_path"))
        summary=[]
        for (m,feature) in sorted({(r["modality"],r["feature"]) for r in comparisons}):
            x=[r for r in comparisons if r["modality"]==m and r["feature"]==feature];a=np.asarray([r["absolute_difference"] for r in x])
            summary.append({"modality":m,"feature":feature,"n_compared":len(x),"n_equal_or_close":sum(r["equal_or_close"] for r in x),"mean_absolute_difference":a.mean(),"median_absolute_difference":np.median(a),"max_absolute_difference":a.max()})
        write_dynamic(metadata/"recording_vs_trial_features_summary.csv",summary,("modality","feature","n_compared","n_equal_or_close","mean_absolute_difference","median_absolute_difference","max_absolute_difference"))
        log_processing(metadata/"processing_log.csv","MULTIMODALPHYSIOKIT_RECORDING_PROCESS_PHASES",Path(__file__).name,"metadata/multimodalphysiokit_recording_processing_manifest.csv","success",f"recordings={len(recording_manifest)}; phases={len(validation)}; processing_rows={len(processing)}; old_outputs_modified=false")
        for stage in (rec_plot_root/".stage",proc_plot_root/".stage"):
            if stage.exists():
                for directory in sorted((p for p in stage.glob("**/*") if p.is_dir()),key=lambda p:len(p.parts),reverse=True):
                    directory.rmdir()
                stage.rmdir()
    print(f"recordings={len(recording_manifest)} phases={len(validation)} processing_rows={len(processing)}")
    return 0
if __name__=="__main__":raise SystemExit(main())
