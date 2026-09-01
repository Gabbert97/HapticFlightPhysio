#!/usr/bin/env python3
"""Build current-state visual diagnostics for processing and coverage failures."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from multimodalphysiokit.io import read_biosignalsplux_hdf5
from multimodalphysiokit.processors.respiration import (
    _definitive_breath_points, _filter_respiration_signal, _find_peaks_respiration,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dataset.provenance import log_processing  # noqa: E402
from processing.common import read_csv_rows, segment_signal  # noqa: E402

REVIEW_FIELDS = (
    "participant_id", "group", "phase", "trial_number", "modality",
    "processing_status", "failure_type", "exclusion_reason", "source_hdf5",
    "start_sample", "end_sample", "diagnostic_plot", "diagnostic_level",
    "exception_type", "exception_message", "traceback_location",
    "manual_review_status", "manual_review_decision", "manual_review_notes",
)
SUMMARY_FIELDS = ("modality", "failure_type", "participant_id", "count")


def save_or_show(fig, path: Path, *, show: bool, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        plt.close(fig); return
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    if show: plt.show()
    plt.close(fig)


def eda_plot(case: dict[str, str], qc: dict[str, str], trials: list[dict[str, str]],
             path: Path, show: bool, overwrite: bool) -> None:
    recording = read_biosignalsplux_hdf5(ROOT / case["source_hdf5"])
    signal = recording.get_signal("eda"); time = np.arange(signal.n_samples) / signal.sampling_frequency
    fig, ax = plt.subplots(figsize=(15, 6), constrained_layout=True)
    ax.plot(time, signal.samples, linewidth=.45, color="tab:blue", label="Converted EDA")
    ceiling=float(qc["eda_conversion_ceiling_us"]); near=float(qc["eda_near_ceiling_threshold_us"])
    ax.axhline(ceiling,color="red",linestyle="--",label=f"ADC ceiling {ceiling:.4f} µS")
    ax.axhline(near,color="darkorange",linestyle="--",label=f"Near ceiling {near:.4f} µS")
    ax.axhline(20,color="purple",linestyle=":",label="20 µS")
    for trial in trials:
        ax.axvline(float(trial["start_offset_s"]),color="black",alpha=.16,linewidth=.7)
        ax.axvline(float(trial["end_offset_s"]),color="black",alpha=.16,linewidth=.7)
    stats=(f"mean={float(qc['eda_mean_us']):.4f} µS\nmedian={float(qc['eda_median_us']):.4f} µS\n"
           f"SD={float(qc['eda_std_us']):.4f} µS\nat ceiling={float(qc['fraction_at_ceiling']):.3%}\n"
           f"near ceiling={float(qc['fraction_near_ceiling']):.3%}\n>=20 µS={float(qc['fraction_at_or_above_20us']):.3%}\n"
           f"classification={qc['eda_saturation_status']}")
    ax.text(.01,.98,stats,transform=ax.transAxes,va="top",bbox=dict(facecolor="white",alpha=.9))
    ax.set(title=f"{case['participant_id']} — {case['phase']} — EDA\nEXCLUDED: SATURATED EDA",
           xlabel="Time [s]",ylabel="EDA [µS]"); ax.legend(loc="lower right")
    save_or_show(fig,path,show=show,overwrite=overwrite)


def coverage_plot(trial: dict[str, str], path: Path, show: bool, overwrite: bool) -> None:
    p0,p1=0.0,float(trial["physio_duration_s"]); b0=float(trial["start_offset_s"]); b1=float(trial["end_offset_s"])
    overlap0=max(p0,b0); overlap1=min(p1,b1); overlap=max(0.0,overlap1-overlap0)
    left=min(p0,b0)-10; right=max(p1,b1)+10
    fig,ax=plt.subplots(figsize=(14,4.5),constrained_layout=True)
    ax.broken_barh([(p0,p1-p0)],(20,8),facecolors="tab:blue",label="Physiological recording")
    ax.broken_barh([(b0,b1-b0)],(8,8),facecolors="tab:orange",label="Requested behavioral trial")
    if overlap>0: ax.broken_barh([(overlap0,overlap)],(8,8),facecolors="tab:green",label="Available overlap")
    ax.axvline(p0,color="tab:blue",linestyle="--");ax.axvline(p1,color="tab:blue",linestyle="--")
    ax.axvline(b0,color="tab:orange",linestyle=":");ax.axvline(b1,color="tab:orange",linestyle=":")
    missing=float(trial["behavior_duration_s"])-overlap
    annotation=(f"alignment={trial['alignment_status']}  trial={float(trial['behavior_duration_s']):.3f}s  "
                f"overlap={overlap:.3f}s  coverage={float(trial['coverage_percent']):.3f}%  missing={missing:.3f}s")
    ax.text(.01,.96,annotation,transform=ax.transAxes,va="top",bbox=dict(facecolor="white",alpha=.9))
    ax.set_xlim(left,right);ax.set_yticks([12,24],labels=["Behavior","Physiology"]);ax.set_xlabel("Seconds relative to physiological recording start")
    ax.set_title(f"{trial['participant_id']} — {trial['phase']} — Trial {trial['trial_number_chronological']}\nEXCLUDED: {trial['alignment_status'].upper()} PHYSIOLOGICAL COVERAGE")
    ax.legend(loc="lower right");save_or_show(fig,path,show=show,overwrite=overwrite)


def respiration_plot(case: dict[str, str], path: Path, show: bool, overwrite: bool) -> None:
    recording=read_biosignalsplux_hdf5(ROOT/case["source_hdf5"])
    raw=segment_signal(recording.get_signal("rip"),int(case["start_sample"]),int(case["end_sample"]))
    factor=int(round(raw.sampling_frequency/100)); down=raw.samples[::factor]; time=np.arange(down.size)/100
    filtered=_filter_respiration_signal(down); minima,maxima=_find_peaks_respiration(filtered)
    accepted_min=np.array([],dtype=int);accepted_max=np.array([],dtype=int)
    try: accepted_min,accepted_max=_definitive_breath_points(filtered,time,minima,maxima)
    except ValueError: pass
    fig,axes=plt.subplots(2,1,sharex=True,figsize=(15,7),constrained_layout=True)
    axes[0].plot(np.arange(raw.n_samples)/raw.sampling_frequency,raw.samples,linewidth=.55);axes[0].set_ylabel(f"Raw RESP [{raw.units}]")
    axes[1].plot(time,filtered,linewidth=.8,label="Package-filtered RESP")
    axes[1].scatter(time[minima],filtered[minima],s=10,color="tab:blue",label=f"candidate minima ({len(minima)})")
    axes[1].scatter(time[maxima],filtered[maxima],s=10,color="tab:orange",label=f"candidate maxima ({len(maxima)})")
    if accepted_min.size: axes[1].scatter(time[accepted_min],filtered[accepted_min],s=22,color="green",label=f"accepted minima ({len(accepted_min)})")
    if accepted_max.size: axes[1].scatter(time[accepted_max],filtered[accepted_max],s=22,color="red",label=f"accepted maxima ({len(accepted_max)})")
    axes[1].legend(loc="best",ncol=2);axes[1].set(xlabel="Time within trial [s]",ylabel="Processed RESP")
    fig.suptitle(f"{case['participant_id']} — {case['phase']} — Trial {case['trial_number']} — RESP\nFAILED: {case['error_message']}")
    save_or_show(fig,path,show=show,overwrite=overwrite)


def build_cases():
    manifest=read_csv_rows(ROOT/"metadata/physiological_processing_manifest.csv")
    trials=read_csv_rows(ROOT/"metadata/trial_index.csv"); qcrows=read_csv_rows(ROOT/"metadata/eda_saturation_qc.csv")
    qc={(r["participant_id"],r["phase"]):r for r in qcrows}; groups={r["participant_id"]:r["group"] for r in manifest if r["group"]}
    rows=[]; plot_jobs={}
    current=[r for r in manifest if r["processing_status"] in {"failed","excluded","skipped","blocked","blocked_methodological"}]
    for case in current:
        modality=case["modality"]; pid=case["participant_id"]; phase=case["phase"]
        if modality=="eda" and case["exclusion_reason"]=="saturated_eda":
            rel=f"outputs/processing_failures/eda/{pid}/{phase}.png"; level="phase"; ftype="qc_exclusion"
            plot_jobs.setdefault(rel,("eda",case,qc[(pid,phase)]))
        else:
            rel=f"outputs/processing_failures/{modality}/{pid}/{phase}_trial_{int(case['trial_number']):02d}.png"
            level="trial";ftype="processing_failure" if case["processing_status"]=="failed" else "qc_exclusion"
            plot_jobs.setdefault(rel,(modality,case,None))
        error=case.get("error_message","");etype,_,emsg=error.partition(": ")
        rows.append({"participant_id":pid,"group":case["group"],"phase":phase,"trial_number":case["trial_number"],"modality":modality,
          "processing_status":case["processing_status"],"failure_type":ftype,"exclusion_reason":case["exclusion_reason"],"source_hdf5":case["source_hdf5"],
          "start_sample":case["start_sample"],"end_sample":case["end_sample"],"diagnostic_plot":rel,"diagnostic_level":level,
          "exception_type":etype if error else "","exception_message":emsg if error else "",
          "traceback_location":"multimodalphysiokit.processors.respiration._extract_respiration_features" if modality=="resp" and error else "",
          "manual_review_status":"pending" if case["processing_status"]=="failed" else "informational","manual_review_decision":"","manual_review_notes":""})
    for trial in trials:
        if trial["include_physio_analysis"].lower()=="true": continue
        pid=trial["participant_id"]; phase=trial["phase"]; number=trial["trial_number_chronological"]
        rel=f"outputs/processing_failures/coverage/{pid}/{phase}_trial_{int(number):02d}.png"
        plot_jobs[rel]=("coverage",trial,None)
        rows.append({"participant_id":pid,"group":groups.get(pid,""),"phase":phase,"trial_number":number,"modality":"coverage",
          "processing_status":"excluded","failure_type":"data_coverage_exclusion","exclusion_reason":trial["physio_exclusion_reason"] or trial["alignment_status"],
          "source_hdf5":trial["physio_filepath"],"start_sample":trial["physio_start_sample"],"end_sample":trial["physio_end_sample"],
          "diagnostic_plot":rel,"diagnostic_level":"trial","exception_type":"","exception_message":"","traceback_location":"",
          "manual_review_status":"informational","manual_review_decision":"","manual_review_notes":""})
    return rows,plot_jobs,trials


def main()->int:
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--all",action="store_true");p.add_argument("--modality",choices=("ecg","eda","resp","temp","fnirs","coverage"));p.add_argument("--subject",type=int);p.add_argument("--phase");p.add_argument("--show-plots",action="store_true");p.add_argument("--overwrite",action="store_true");args=p.parse_args()
    rows,jobs,trials=build_cases(); selected=[]
    for rel,job in jobs.items():
        kind,case,_=job
        if args.modality and kind!=args.modality: continue
        if args.subject and case["participant_id"]!=f"Subject_{args.subject}":continue
        if args.phase and case["phase"]!=args.phase:continue
        selected.append((rel,job))
    root=ROOT/"outputs/processing_failures"
    if args.all and args.overwrite and root.exists(): shutil.rmtree(root)
    if args.all:
        for name in ("ecg", "eda", "resp", "temp", "fnirs", "coverage"):
            (root/name).mkdir(parents=True, exist_ok=True)
    for rel,(kind,case,extra) in selected:
        path=ROOT/rel
        if kind=="eda":
            phase_trials=[r for r in trials if r["participant_id"]==case["participant_id"] and r["phase"]==case["phase"] and r["alignment_valid"].lower()=="true"]
            eda_plot(case,extra,phase_trials,path,args.show_plots,args.overwrite)
        elif kind=="coverage": coverage_plot(case,path,args.show_plots,args.overwrite)
        elif kind=="resp": respiration_plot(case,path,args.show_plots,args.overwrite)
        else: raise ValueError(f"No plot implementation for current failure modality {kind}")
    if args.all:
        metadata=ROOT/"metadata"; review=metadata/"physiological_failure_review.csv"
        with review.open("w",encoding="utf-8",newline="") as f:w=csv.DictWriter(f,fieldnames=REVIEW_FIELDS);w.writeheader();w.writerows(rows)
        counts=Counter((r["modality"],r["failure_type"],r["participant_id"]) for r in rows)
        summary=[{"modality":k[0],"failure_type":k[1],"participant_id":k[2],"count":v} for k,v in sorted(counts.items())]
        with (metadata/"physiological_failure_summary.csv").open("w",encoding="utf-8",newline="") as f:w=csv.DictWriter(f,fieldnames=SUMMARY_FIELDS);w.writeheader();w.writerows(summary)
        modality_counts=Counter(r["modality"] for r in rows);type_counts=Counter(r["failure_type"] for r in rows)
        root.mkdir(parents=True,exist_ok=True)
        readme="# Current physiological processing failures\n\nGenerated from current metadata; historical/resolved failures are excluded.\n\n"
        readme+=f"- Review rows: {len(rows)}\n- Unique diagnostic plots: {len(jobs)}\n\n## By modality\n\n"+"\n".join(f"- {k}: {v}" for k,v in sorted(modality_counts.items()))
        readme+="\n\n## By failure type\n\n"+"\n".join(f"- {k}: {v}" for k,v in sorted(type_counts.items()))
        readme+="\n\n## Affected participants\n\n"+"\n".join(f"- {p}" for p in sorted({r['participant_id'] for r in rows},key=lambda x:int(x.split('_')[1])))+"\n"
        (root/"README.md").write_text(readme,encoding="utf-8")
        log_processing(metadata/"processing_log.csv","PROCESSING_FAILURE_DIAGNOSTICS",Path(__file__).name,
          "metadata/physiological_failure_review.csv","success",f"review_rows={len(rows)}; unique_plots={len(jobs)}; counts={json.dumps(dict(modality_counts),sort_keys=True)}; raw_files_modified=false")
        print(f"Current review rows: {len(rows)}; unique plots: {len(jobs)}")
    else: print(f"Generated/verified {len(selected)} selected diagnostic plot(s)")
    return 0

if __name__=="__main__":raise SystemExit(main())
