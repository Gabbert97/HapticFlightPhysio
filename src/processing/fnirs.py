"""fNIRS processing through MultimodalPhysioKit."""

from pathlib import Path

from multimodalphysiokit.core import Signal
from multimodalphysiokit.processors import FNIRSProcessor


def process_fnirs(red: Signal, infrared: Signal, *, age_years: float,
                  red_reference: float, infrared_reference: float,
                  show_plots: bool = False, save_plots: bool = False,
                  output_dir: Path | None = None):
    """Return package ΔHbO/ΔHbR features using an explicit baseline reference."""
    processor = FNIRSProcessor(age_years=age_years)
    return processor.process(
        red, infrared, baseline_red_value=red_reference,
        baseline_infrared_value=infrared_reference, show_plots=show_plots,
        save_plots=save_plots, output_dir=output_dir, output_label="fnirs",
    )
