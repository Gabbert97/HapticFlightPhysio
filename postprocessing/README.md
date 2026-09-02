# Physiological postprocessing

These notebooks analyze the final physiology–Mental Workload (MWL) dataset created by the private preprocessing workflow. They treat `outputs/final_features/physiology_mwl_analysis_dataset.csv` as an authoritative, read-only input and do not rebuild physiological signals, baseline corrections, or MWL mappings. That participant-level dataset is not distributed in this code-only repository.

## Exploratory and supporting analyses

- `01_dataset_overview.ipynb` — dataset dimensions, coverage, missingness, and numerical sanity checks.
- `01_physiology_mwl_correlations.ipynb` — initial overall exploratory physiology–MWL correlations.
- `02_mwl_exploration.ipynb` — descriptive MWL distributions, trajectories, and variability.
- `02_haptic_noha_physiology_comparison.ipynb` — participant-level nonparametric Haptic-versus-NoHA physiological comparisons by phase.
- `03_physiology_mwl_correlations_by_phase.ipynb` — full-feature phase × group physiology–MWL sensitivity analysis.

## Final manuscript analyses

### 04 — Phase-specific physiology–MWL correlations

`04_physiology_mwl_correlations_reduced_features.ipynb` includes Pre-Test, Test 1, Test 2, Test 3, and Evaluation. Spearman correlations are calculated separately for each phase × group family, and Benjamini–Hochberg false-discovery-rate correction is applied within each such family.

### 05 — Overall group-specific physiology–MWL correlations

`05_physiology_mwl_group_correlations_no_pretest.ipynb` is the primary overall group analysis. It excludes Pre-Test and includes Test 1, Test 2, Test 3, and Evaluation. Spearman correlations are calculated separately for Haptic and NoHA, with Benjamini–Hochberg correction applied separately within each group. The notebook also directly compares Haptic and NoHA coefficients using participant-level cluster bootstrap and participant-label permutation procedures.

Both manuscript notebooks use the same fixed set of 51 physiological features. The set is obtained from the original 79 features by handling three constant ECG features and applying deterministic, outcome-independent redundancy filtering at absolute Spearman correlation ≥ 0.80. MWL, group labels, MWL p-values, and group differences are not used to select features.

All correlation analyses are exploratory. A feature reaching significance in one group but not the other does not establish a significant difference between the group-specific correlations; only the explicit direct comparison in notebook 05 addresses that question.

## Running notebooks

Create the environment from the repository root and start Jupyter:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
jupyter lab
```

The notebooks can be launched from either the repository root or `postprocessing/` because they resolve the repository root with `pathlib`. Generated tables and figures are written below `postprocessing/outputs/`, which is intentionally excluded from Git.
