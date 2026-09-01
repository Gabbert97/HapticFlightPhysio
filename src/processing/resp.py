"""RIP respiration processing through MultimodalPhysioKit."""

from pathlib import Path
from multimodalphysiokit.core import Signal
from multimodalphysiokit.processors import RespirationProcessor


def process_resp(signal: Signal, *, show_plots: bool = False,
                 save_plots: bool = False, output_dir: Path | None = None):
    return RespirationProcessor(
        min_breathing_rate=3.0,
        max_breathing_rate=30.0,
    ).process(signal, show_plots=show_plots,
              save_plots=save_plots, output_dir=output_dir)
