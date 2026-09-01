"""Temperature processing through MultimodalPhysioKit."""

from pathlib import Path
from multimodalphysiokit.core import Signal
from multimodalphysiokit.processors import TemperatureProcessor


def process_temp(signal: Signal, *, show_plots: bool = False,
                 save_plots: bool = False, output_dir: Path | None = None):
    return TemperatureProcessor().process(signal, show_plots=show_plots,
                                          save_plots=save_plots, output_dir=output_dir)

