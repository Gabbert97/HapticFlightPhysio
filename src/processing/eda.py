"""EDA processing through MultimodalPhysioKit and cvxEDA 1.1.0."""

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from multimodalphysiokit.core import Signal
from multimodalphysiokit.processors import EDAProcessor


def process_eda(signal: Signal, *, show_plots: bool = False,
                save_plots: bool = False, output_dir: Path | None = None):
    processor = EDAProcessor(minimum_scr_amplitude=0.01)
    if show_plots:
        return processor.process(signal, show_plots=True, save_plots=save_plots,
                                 output_dir=output_dir, output_label="eda")
    # cvxEDA 1.1.0 resets cvxopt's global options to its own verbose defaults.
    # Suppress only that third-party console output; computation and parameters
    # are unchanged and exceptions still propagate to the caller.
    with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
        return processor.process(signal, show_plots=False, save_plots=save_plots,
                                 output_dir=output_dir, output_label="eda")
