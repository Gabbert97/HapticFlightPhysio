# HelicopterHapticTrainingPhysio
Processing codes for physiological signals acquired during a helicopter simulatior training

## Raw physiological alignment validation

The raw-alignment plot visually checks that behavioral trial timestamps map to
the expected locations in each continuous BioSignalsPlux HDF5 phase recording.
Only trials with complete physiological coverage are included by default.
Boundaries come from behavioral `TimeStamp` values already mapped to validated
HDF5 sample indices in `metadata/trial_index.csv`.

The plots contain the complete ECG, EDA, respiration, temperature, fNIRS red
current, and fNIRS infrared current signals converted from ADC values to
physical units by MultimodalPhysioKit, with no
subsequent preprocessing. Experimental-phase black dashed boundaries come from
the included behavioral trials. Baseline black dashed markers identify only
observed 0-to-1 transitions in the HDF5 digital channel; they are point events,
not inferred baseline intervals. No filtering, normalization,
resampling, artifact correction, segmentation, or other signal preprocessing is
applied at this stage.

```bash
python scripts/04_plot_raw_trial_alignment.py \
    --subject 12 \
    --phase test_2
```

Add `--show-excluded` for diagnostic red dotted excluded boundaries. Add
`--save` to write a PNG under `outputs/alignment_plots/`; otherwise the figure
is displayed interactively. Existing PNG files are not replaced unless
`--overwrite` is also supplied.

Generate every available participant and phase plot in batch mode with:

```bash
python scripts/04_plot_raw_trial_alignment.py --all
```

Batch plots are written as `outputs/alignment_plots/Subject_<id>/<phase>.png`.

## ECG polarity manual-review QC

Prepare participant-level automatic polarity classifications and generate an
exactly 30-second raw ECG review plot only for inverted or uncertain cases:

```bash
python scripts/05_ecg_polarity_qc.py --overwrite
```

The displayed waveform is the original MultimodalPhysioKit ADC-to-mV output.
Temporary 5--25 Hz filtering is used only for polarity-invariant QRS detection,
classification, and objective window selection. No multiplier or ECG correction
is applied. Add `--show-qrs-markers` to overlay the temporary detector locations.

Finalize reviewed ECG polarity decisions directly or interactively with:

```bash
python scripts/06_finalize_ecg_polarity.py --subject 2 --decision inverted
python scripts/06_finalize_ecg_polarity.py
```

Final multipliers remain metadata only. The helper functions
`get_ecg_multiplier` and `apply_ecg_polarity` perform strict lookup and optional
in-memory correction without modifying HDF5 files.

## Physiological trial processing

The processing entry point uses only trials already marked
`include_physio_analysis=True` in `metadata/trial_index.csv`. ECG polarity is
corrected in memory using the finalized participant multiplier. EDA eligibility
is phase-specific: `possible_saturation` recordings are retained with a QC
warning, while definitively saturated recordings are excluded from EDA only.
Each other modality continues independently.

Inspect one trial interactively (package diagnostic plots are enabled where
the processor supports them):

```bash
python scripts/07_process_physiological_signals.py \
    --subject 17 --phase evaluation --trial 1 \
    --modality all --show-plots --save-plots
```

Run the complete eligible dataset non-interactively with:

```bash
python scripts/07_process_physiological_signals.py --all --overwrite
```

Batch outputs are modality-specific feature tables and
`metadata/physiological_processing_manifest.csv`; raw or processed waveform
copies are not written. Baseline remains continuous QC-only because its digital
markers are point events, not trial intervals. For fNIRS, the first baseline
digital rising edge is used only as an anchor: the participant reference window
starts 30 seconds later and targets 180 seconds; a shortened window is accepted
only when more than 120 seconds remain. MultimodalPhysioKit
estimates filtered-current reference means from that window and uses the
participant's workbook `Age` for wavelength-specific DPF and modified
Beer–Lambert ΔHbO/ΔHbR processing. Invalid windows or missing ages exclude
only that participant's fNIRS trials.

Final RESP extraction uses a validated accepted respiratory-rate range of
3–30 breaths/min consistently for baseline and experimental observations.

Build and inspect participant fNIRS references with:

```bash
python scripts/08_build_fnirs_baseline_reference.py --all --save-plots
python scripts/08_build_fnirs_baseline_reference.py --subject 17 --show-plots
```

## Processing-failure review

Regenerate current-state failure diagnostics after processing with:

```bash
python scripts/09_build_processing_failure_plots.py --all --overwrite
```

The review consolidates deterministic phase-level exclusions, such as EDA
saturation, while retaining one metadata row for every affected trial. Trial
specific processing and physiological-coverage failures receive individual
plots. Historical failures that have subsequently recovered are not included.
Use `--modality`, `--subject`, `--phase`, and `--show-plots` for focused manual
inspection.

## MultimodalPhysioKit native debug plots

Generate every figure exposed by the installed package's production processor
methods, without repository-defined plotting, using:

```bash
python scripts/10_multimodalphysiokit_debug_plots.py --all --save-plots --overwrite
```

Outputs use flat modality folders under `outputs/multimodalphysiokit_debug/`,
with participant, phase, and trial encoded in each filename. The corresponding
manifest contains one row per package-native figure. Focused runs support
`--subject`, `--phase`, `--trial`, `--modality`, and `--show-plots`.

## MultimodalPhysioKit Recording validation

The independent Recording-based validation constructs one Recording per HDF5,
maps included behavioral trials to named phases, validates native phase indexes,
and extracts features via `Recording.process_phases()`:

```bash
python scripts/12_process_multimodalphysiokit_recordings.py \
    --all --save-plots --overwrite
```

All Recording feature tables, comparisons, manifests, and native figures use
dedicated `recording_*` metadata names and
`outputs/multimodalphysiokit_recording*` directories. Existing trial-level
outputs are not replaced.

## ECG STFT validation experiment

The independent mode-1 ECG experiment tests the package's native 60-second
STFT path without changing the production mode-2 configuration:

```bash
python scripts/13_validate_ecg_stft.py --all --save-plots --overwrite
```

Results are isolated in `metadata/ecg_stft_validation.csv` and
`outputs/ecg_stft_validation/`.

## Final physiological feature dataset

The authoritative analysis-ready dataset is
`outputs/final_features/physiological_features_all.csv`. It contains 21
participants and 810 observations: 21 participant-level baseline observations
and 789 experimental trials. Its 79 physiological features cover ECG, EDA,
RESP, skin temperature, and fNIRS. All available modalities for the same
participant/phase/trial are stored on one row; a modality-specific exclusion or
failure remains missing without removing the rest of that observation.

The common baseline interval starts 30 seconds after the first baseline digital
rising-edge marker and targets 180 seconds. A shortened interval is accepted
only when more than 120 seconds remain. Final RESP features use the validated
3–30 breaths/min acceptance range. Experimental fNIRS ΔHbO/ΔHbR features use
each participant's own baseline-derived Red/Infrared reference values; trials
do not estimate their own reference.

Feature definitions and known units are documented in
`outputs/final_features/feature_dictionary.csv`.
