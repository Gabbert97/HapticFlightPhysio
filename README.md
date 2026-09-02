# HelicopterHapticTrainingPhysio

This repository contains the Python processing and statistical-analysis code used to analyze multimodal physiological recordings collected during a helicopter roll-tracking simulator training experiment comparing haptic-feedback training with training without haptic feedback (NoHA). The analyzed modalities are electrocardiography (ECG), electrodermal activity (EDA), respiration, skin temperature, and functional near-infrared spectroscopy (fNIRS).

## Associated study

This code supports the manuscript **“Effects of Full-Body Haptic Feedback on Training Performance and Mental Workload in a Helicopter Roll-Tracking Task.”** Bibliographic details will be added when they become available.

## Repository structure

- `scripts/` — data organization, alignment, quality control, physiological processing, feature extraction, and final dataset construction.
- `src/` — reusable dataset and modality-processing utilities.
- `postprocessing/` — exploratory statistical analyses and the analyses used for the manuscript.

Private study data, participant-level metadata, generated feature tables, and analysis outputs are intentionally not distributed in this repository.

## Installation

The development environment used Python 3.14. Create an isolated environment and install the declared dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The processing code depends on [MultimodalPhysioKit](https://github.com/Gabbert97/MultimodalPhysioKit). The exact public release used by the final workflow is pinned in `requirements.txt`; it provides the configurable `RespirationProcessor` rate limits used for the validated 3–30 breaths/min respiration analysis.

## Data availability

This is a code-only research repository. It does not include:

- raw physiological recordings;
- behavioral participant-level data;
- participant metadata or demographic information;
- participant-level Mental Workload (MWL) ratings;
- intermediate or final participant-level feature tables.

These materials are intentionally excluded to protect participant privacy and comply with the study’s data-sharing constraints. The scripts expect an authorized local `data/`, `metadata/`, and `outputs/` structure; a fresh clone cannot reproduce participant-level results without those non-public inputs.

## Physiological processing

MultimodalPhysioKit provides the modality-specific signal-processing implementation. The final analysis representation contains 79 physiological features across ECG, EDA, respiration, skin temperature, and fNIRS. Experimental fNIRS features use participant-specific baseline-derived optical references. ECG, EDA, respiration, and temperature features follow the study’s participant-level baseline-processing and correction workflow.

ECG polarity correction is applied in memory, EDA exclusions remain modality-specific, and the final respiration acceptance range is 3–30 breaths/min. Raw recordings are never overwritten by the processing workflow.

## Processing workflow

The numbered scripts reflect the historical implementation order; some are optional QC or validation utilities rather than mandatory production steps.

- `01`–`06` — acquisition-file organization, behavioral/physiological alignment, and ECG/EDA quality control.
- `07`–`10` — physiological processing, baseline-reference construction, failure review, and package-native diagnostic validation.
- `12`–`15` — MultimodalPhysioKit Recording/STFT validation, baseline feature extraction, and final 3–30 breaths/min RESP processing.
- `16` — construction of the final wide physiological feature dataset.
- `17` — mapping of subjective MWL observations to physiological blocks.

`scripts/03_cleanup_empty_behavioral_files.py` is a **destructive cleanup utility**: it deletes behavioral files classified as invalid or empty. Do not run it without reviewing its selection logic and the targeted files. See the script help and source before executing any data-mutating utility.

## Manuscript analyses

The two final manuscript-analysis notebooks use the same fixed physiological feature set:

- `postprocessing/04_physiology_mwl_correlations_reduced_features.ipynb` — phase-specific physiology–MWL correlations for Pre-Test, Test 1, Test 2, Test 3, and Evaluation, calculated separately by group and phase.
- `postprocessing/05_physiology_mwl_group_correlations_no_pretest.ipynb` — primary overall Haptic and NoHA physiology–MWL analysis pooling Test 1, Test 2, Test 3, and Evaluation while excluding Pre-Test; it also performs a participant-level direct comparison of group-specific correlations.

The analyses begin with 79 physiological features, exclude constant or unusable features, and apply deterministic redundancy filtering at absolute Spearman correlation ≥ 0.80. This reduction is independent of MWL and group labels and yields a common set of 51 features. Physiology–MWL associations use Spearman correlations with Benjamini–Hochberg false-discovery-rate correction in the families defined by each notebook. Significance in one group but not another is not, by itself, evidence of a significant group difference.

## Reproducibility

A public clone provides the processing and analysis implementation, not the private observations required to reproduce participant-level numerical results. The notebooks locate the repository root automatically but require the non-public final analysis dataset at the documented local path. The publicly released and pinned MultimodalPhysioKit dependency provides the exact configurable respiratory-rate API used by the final workflow.

## License

Repository code is released under the [MIT License](LICENSE). Third-party packages retain their respective licenses.

## Citation

If you use this code, please cite the associated manuscript once bibliographic details become available.

## Author

Gabriele Luzzani
