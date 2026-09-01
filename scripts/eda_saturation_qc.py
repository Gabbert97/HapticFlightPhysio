#!/usr/bin/env python3
"""Classify participant-phase EDA saturation before physiological processing.

The MultimodalPhysioKit BioSignalsPlux conversion is
``ADC * 3 / (0.12 * 2**16)``, giving a maximum converted value of
24.999618530273438 uS. Classification uses the converted EDA directly.

Rules (all thresholds are explicit and reproducible):

* saturated: at least 10% of samples equal the converted ADC ceiling, OR at
  least 30% are within 0.5 uS of the ceiling while SD <= 1.0 uS;
* possible_saturation: not saturated, but at least 1% are within 0.5 uS of the
  ceiling, OR at least 25% are >= 20 uS and the recording reaches within
  0.5 uS of the ceiling;
* ok: neither rule is met.

Thus a brief isolated maximum does not by itself trigger exclusion or review.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from multimodalphysiokit.io import read_biosignalsplux_hdf5

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from dataset.naming import CANONICAL_PHASES  # noqa: E402
from dataset.provenance import log_processing  # noqa: E402

ADC_BITS = 16
EDA_CEILING_US = (2**ADC_BITS - 1) * 3.0 / (0.12 * 2**ADC_BITS)
NEAR_CEILING_US = EDA_CEILING_US - 0.5
HIGH_US = 20.0
FIELDS = [
    "participant_id", "participant_numeric_id", "phase", "source_hdf5",
    "sampling_rate_hz", "n_samples", "eda_conversion_ceiling_us",
    "eda_near_ceiling_threshold_us", "eda_min_us", "eda_max_us", "eda_mean_us",
    "eda_median_us", "eda_std_us", "eda_dynamic_range_us", "eda_q01_us",
    "eda_q99_us", "fraction_at_ceiling", "fraction_near_ceiling",
    "fraction_at_or_above_20us", "eda_saturation_status",
    "eda_include_for_processing", "manual_review_required", "manual_review_status",
    "diagnostic_plot", "classification_rule", "notes",
]


def classify(samples: np.ndarray) -> tuple[str, dict[str, float]]:
    tolerance = 1e-12
    metrics = {
        "min": float(np.min(samples)), "max": float(np.max(samples)),
        "mean": float(np.mean(samples)), "median": float(np.median(samples)),
        "std": float(np.std(samples)), "range": float(np.ptp(samples)),
        "q01": float(np.quantile(samples, 0.01)),
        "q99": float(np.quantile(samples, 0.99)),
        "at_ceiling": float(np.mean(samples >= EDA_CEILING_US - tolerance)),
        "near_ceiling": float(np.mean(samples >= NEAR_CEILING_US)),
        "above_20": float(np.mean(samples >= HIGH_US)),
    }
    saturated = (
        metrics["at_ceiling"] >= 0.10
        or (metrics["near_ceiling"] >= 0.30 and metrics["std"] <= 1.0)
    )
    possible = (
        metrics["near_ceiling"] >= 0.01
        or (metrics["above_20"] >= 0.25 and metrics["max"] >= NEAR_CEILING_US)
    )
    return ("saturated" if saturated else "possible_saturation" if possible else "ok"), metrics


def plot_diagnostic(samples: np.ndarray, sampling_rate: float, participant: str,
                    phase: str, status: str, output: Path, overwrite: bool) -> None:
    if output.exists() and not overwrite:
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    # Plot every sample; rasterization limits vector-rendering overhead without
    # resampling or changing the values used by QC.
    time = np.arange(samples.size) / sampling_rate
    fig, axis = plt.subplots(figsize=(16, 6), constrained_layout=True)
    axis.plot(time, samples, linewidth=0.45, color="tab:blue", rasterized=True)
    axis.axhline(EDA_CEILING_US, color="red", linestyle="--", linewidth=1,
                label=f"ADC conversion ceiling ({EDA_CEILING_US:.4f} µS)")
    axis.axhline(NEAR_CEILING_US, color="darkorange", linestyle=":", linewidth=1,
                label=f"Near-ceiling threshold ({NEAR_CEILING_US:.4f} µS)")
    axis.axhline(HIGH_US, color="gray", linestyle=":", linewidth=1, label="20 µS")
    axis.set_title(f"{participant.replace('_', ' ')} — {phase} — EDA saturation QC: {status}")
    axis.set_xlabel("Time from physiological recording start [s]")
    axis.set_ylabel("EDA [µS]")
    axis.legend(loc="best")
    axis.grid(True, alpha=0.2)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--metadata-dir", type=Path, default=REPOSITORY_ROOT / "metadata")
    parser.add_argument("--output-dir", type=Path,
                        default=REPOSITORY_ROOT / "outputs/qc/eda_saturation")
    args = parser.parse_args()
    with (args.metadata_dir / "physiological_recordings.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        physio = list(csv.DictReader(handle))
    physio.sort(key=lambda row: (
        int(row["participant_numeric_id"]), CANONICAL_PHASES.index(row["phase"])
    ))
    rows = []
    for source in physio:
        path = REPOSITORY_ROOT / source["physio_filepath"]
        signal = read_biosignalsplux_hdf5(path).get_signal("eda")
        if signal.n_samples != int(source["n_samples"]):
            raise ValueError(f"EDA sample-count disagreement for {path}")
        if signal.sampling_frequency != float(source["sampling_rate_hz"]):
            raise ValueError(f"EDA sampling-rate disagreement for {path}")
        status, metrics = classify(signal.samples)
        review = status == "possible_saturation"
        include = status == "ok"
        plot_relative = ""
        if status != "ok":
            plot_path = args.output_dir / source["participant_id"] / f"{source['phase']}.png"
            plot_diagnostic(signal.samples, signal.sampling_frequency,
                            source["participant_id"], source["phase"], status,
                            plot_path, args.overwrite)
            plot_relative = plot_path.relative_to(REPOSITORY_ROOT).as_posix()
        rows.append({
            "participant_id": source["participant_id"],
            "participant_numeric_id": source["participant_numeric_id"],
            "phase": source["phase"], "source_hdf5": source["physio_filepath"],
            "sampling_rate_hz": f"{signal.sampling_frequency:g}",
            "n_samples": signal.n_samples,
            "eda_conversion_ceiling_us": f"{EDA_CEILING_US:.9f}",
            "eda_near_ceiling_threshold_us": f"{NEAR_CEILING_US:.9f}",
            "eda_min_us": f"{metrics['min']:.6f}", "eda_max_us": f"{metrics['max']:.6f}",
            "eda_mean_us": f"{metrics['mean']:.6f}", "eda_median_us": f"{metrics['median']:.6f}",
            "eda_std_us": f"{metrics['std']:.6f}", "eda_dynamic_range_us": f"{metrics['range']:.6f}",
            "eda_q01_us": f"{metrics['q01']:.6f}", "eda_q99_us": f"{metrics['q99']:.6f}",
            "fraction_at_ceiling": f"{metrics['at_ceiling']:.9f}",
            "fraction_near_ceiling": f"{metrics['near_ceiling']:.9f}",
            "fraction_at_or_above_20us": f"{metrics['above_20']:.9f}",
            "eda_saturation_status": status,
            "eda_include_for_processing": str(include),
            "manual_review_required": str(review),
            "manual_review_status": "pending" if review else "not_required",
            "diagnostic_plot": plot_relative,
            "classification_rule": "ceiling=24.999618530uS; saturated: ceiling>=10% or near>=30%+SD<=1; possible: near>=1% or >=20uS>=25%+reaches near",
            "notes": "Do not process EDA until manually resolved" if review else (
                "Excluded from EDA processing" if status == "saturated" else ""
            ),
        })
        print(f"{source['participant_id']}/{source['phase']}: {status}; "
              f"near={metrics['near_ceiling']:.3%}; ceiling={metrics['at_ceiling']:.3%}; "
              f"std={metrics['std']:.3f} µS")
    target = args.metadata_dir / "eda_saturation_qc.csv"
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    counts = {status: sum(row["eda_saturation_status"] == status for row in rows)
              for status in ("ok", "possible_saturation", "saturated")}
    pending = [f"{row['participant_id']}/{row['phase']}" for row in rows
               if row["manual_review_status"] == "pending"]
    log_processing(
        args.metadata_dir / "processing_log.csv", "EDA_SATURATION_QC",
        Path(__file__).name, "metadata/eda_saturation_qc.csv",
        "manual_review_required" if pending else "success",
        f"ceiling_us={EDA_CEILING_US:.9f}; ok={counts['ok']}; "
        f"possible_saturation={counts['possible_saturation']}; saturated={counts['saturated']}; "
        f"pending={','.join(pending)}; eda_processing_started=false",
    )
    print(f"Counts: {counts}")
    if pending:
        print("STOP: manual review required for " + ", ".join(pending))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
