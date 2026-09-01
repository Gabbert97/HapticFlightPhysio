#!/usr/bin/env python3
"""Independent Recording-based ECG mode-1 STFT validation experiment."""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import multimodalphysiokit
import numpy as np
from multimodalphysiokit.core import Signal
from multimodalphysiokit.io import read_biosignalsplux_hdf5
from multimodalphysiokit.processors import ECGProcessor

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from dataset.ecg_polarity import get_ecg_multiplier  # noqa:E402
from processing.common import read_csv_rows  # noqa:E402

FIELDS=("participant_id","phase","trial_number","duration_s","n_rpeaks","n_rr_intervals","stft_status","stft_matrix_shape","stft_power_shape","stft_frequencies_shape","stft_times_shape","n_stft_time_windows","n_stft_frequency_bins","spectrogram_generated","spectrogram_path","native_plot_paths","failure_reason")

def truth(value):return value.strip().lower()=="true"

def recording_for_trial(trial,ecg_config):
    rec=read_biosignalsplux_hdf5(ROOT/trial["physio_filepath"],name=f"{trial['participant_id']}_{trial['phase']}")
    ecg=rec.get_signal("ecg");mult=get_ecg_multiplier(trial["participant_id"],ecg_config)
    rec.add_signal("ecg",Signal(samples=ecg.samples*mult,time=ecg.time.copy(),sampling_frequency=ecg.sampling_frequency,label="ecg_corrected",units=ecg.units,metadata={**ecg.metadata,"ecg_multiplier_final":mult,"polarity_correction":"in_memory"}))
    fs=ecg.sampling_frequency;start=int(trial["physio_start_sample"]);end=int(trial["physio_end_sample"])
    label=f"trial_{int(trial['trial_number_chronological']):02d}";rec.set_phases(np.asarray([[start/fs],[end/fs]]),[label])
    return rec,label,fs,start,end

def move_native(stage,destination,prefix,overwrite):
    paths=[]
    for source in sorted(stage.glob("**/*.png")):
        target=destination/f"{prefix}_{source.name}"
        if target.exists() and not overwrite:raise FileExistsError(target)
        destination.mkdir(parents=True,exist_ok=True);source.replace(target);paths.append(target)
    if stage.exists():shutil.rmtree(stage)
    return paths

def run_trial(trial,output,ecg_config,prior_frequency,show,save,overwrite):
    rec,label,fs,start,end=recording_for_trial(trial,ecg_config);processor=ECGProcessor(label_frequency_analysis=1,return_intermediates=True)
    prefix=f"{trial['participant_id']}_{trial['phase']}_{label}";stage=output/".stage"/prefix
    kwargs={"show_plots":show,"save_plots":save,"output_dir":stage,"output_label":"ecg"}
    result=None;failure="";before=set(plt.get_fignums())
    try:result=rec.process_phases(processor,"ecg",phases=[label],process_kwargs=kwargs)[label]
    except Exception as exc:failure=f"{type(exc).__name__}: {exc}"
    finally:
        for number in set(plt.get_fignums())-before:plt.close(number)
    paths=move_native(stage,output,prefix,overwrite) if save else []
    prior=prior_frequency[(trial["participant_id"],trial["phase"],str(int(trial["trial_number_chronological"])))]
    row={"participant_id":trial["participant_id"],"phase":trial["phase"],"trial_number":trial["trial_number_chronological"],"duration_s":(end-start)/fs,"n_rpeaks":prior["n_rpeaks"],"n_rr_intervals":prior["n_rr_intervals"],"stft_status":"success" if result else "failed","stft_matrix_shape":"","stft_power_shape":"","stft_frequencies_shape":"","stft_times_shape":"","n_stft_time_windows":"","n_stft_frequency_bins":"","spectrogram_generated":False,"spectrogram_path":"","native_plot_paths":";".join(p.relative_to(ROOT).as_posix() for p in paths),"failure_reason":failure}
    if result:
        inter=result.intermediates
        required=("stft_matrix","stft_power","stft_frequencies","stft_times")
        if not all(name in inter for name in required):raise RuntimeError(f"Successful mode-1 result lacks STFT intermediates: {required}")
        matrix=np.asarray(inter["stft_matrix"]);power=np.asarray(inter["stft_power"]);freq=np.asarray(inter["stft_frequencies"]);times=np.asarray(inter["stft_times"])
        spectrograms=[p for p in paths if p.name.endswith("ecg_STFT_HRV.png")]
        row.update(stft_matrix_shape="x".join(map(str,matrix.shape)),stft_power_shape="x".join(map(str,power.shape)),stft_frequencies_shape="x".join(map(str,freq.shape)),stft_times_shape="x".join(map(str,times.shape)),n_stft_time_windows=matrix.shape[1],n_stft_frequency_bins=matrix.shape[0],spectrogram_generated=bool(spectrograms),spectrogram_path=";".join(p.relative_to(ROOT).as_posix() for p in spectrograms))
    return row

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--all",action="store_true");p.add_argument("--subject",type=int);p.add_argument("--phase");p.add_argument("--trial",type=int);p.add_argument("--show-plots",action="store_true");p.add_argument("--save-plots",action="store_true");p.add_argument("--overwrite",action="store_true");args=p.parse_args()
    if not args.all and args.subject is None:p.error("use --all or --subject")
    if args.trial is not None and args.phase is None:p.error("--trial requires --phase")
    if args.all and args.show_plots:p.error("--show-plots is not allowed with --all")
    output=ROOT/"outputs/ecg_stft_validation";metadata=ROOT/"metadata";csv_path=metadata/"ecg_stft_validation.csv"
    if args.all and not args.overwrite and (csv_path.exists() or output.exists()):raise FileExistsError("STFT validation outputs exist; use --overwrite")
    if args.all and args.overwrite and output.exists():shutil.rmtree(output)
    output.mkdir(parents=True,exist_ok=True)
    trials=[r for r in read_csv_rows(metadata/"trial_index.csv") if truth(r["include_physio_analysis"])]
    if args.subject is not None:trials=[r for r in trials if r["participant_id"]==f"Subject_{args.subject}"]
    if args.phase:trials=[r for r in trials if r["phase"]==args.phase]
    if args.trial is not None:trials=[r for r in trials if int(r["trial_number_chronological"])==args.trial]
    prior={(r["participant_id"],r["phase"],str(int(r["trial_number"]))):r for r in read_csv_rows(metadata/"recording_ecg_frequency_validation.csv")}
    rows=[]
    for i,trial in enumerate(trials,1):
        rows.append(run_trial(trial,output,metadata/"ecg_polarity_qc.csv",prior,args.show_plots,args.save_plots,args.overwrite))
        if args.all and i%50==0:print(f"STFT validation {i}/{len(trials)}")
    if args.all:
        with csv_path.open("w",newline="",encoding="utf-8") as f:w=csv.DictWriter(f,fieldnames=FIELDS);w.writeheader();w.writerows(rows)
    for row in rows if not args.all else []:print(row)
    print(f"trials={len(rows)} success={sum(r['stft_status']=='success' for r in rows)} failed={sum(r['stft_status']=='failed' for r in rows)} spectrograms={sum(bool(r['spectrogram_generated']) for r in rows)}")
    return 0
if __name__=="__main__":raise SystemExit(main())
