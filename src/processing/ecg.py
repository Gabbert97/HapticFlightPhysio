"""ECG processing through MultimodalPhysioKit."""

from pathlib import Path
import numpy as np
from multimodalphysiokit.core import Signal
from multimodalphysiokit.processors import ECGProcessor


def process_ecg(signal: Signal, multiplier: int, *, show_plots: bool = False,
                save_plots: bool = False, output_dir: Path | None = None):
    if multiplier not in (-1, 1):
        raise ValueError("ECG multiplier must be +1 or -1")
    corrected = Signal(samples=signal.samples * multiplier, time=signal.time.copy(),
                       sampling_frequency=signal.sampling_frequency, label="ecg_corrected",
                       units=signal.units, metadata={**signal.metadata,
                       "ecg_multiplier_applied": multiplier})
    processor = ECGProcessor(label_frequency_analysis=2, return_intermediates=True)
    if show_plots or save_plots:
        processor.detect_r_peaks(corrected, show_plots=show_plots,
                                 save_plots=save_plots, output_dir=output_dir)
    result = processor.process(corrected, show_plots=show_plots, save_plots=save_plots,
                               output_dir=output_dir, output_label="ecg")
    result.metadata["ecg_multiplier_applied"] = multiplier
    result.metadata["timing_preserved"] = bool(np.array_equal(corrected.time, signal.time))
    return result

