# Physiological postprocessing

Physiological preprocessing, quality control, baseline correction, trial aggregation, and MWL mapping are finalized elsewhere in this repository. The notebooks here treat [`outputs/final_features/physiology_mwl_analysis_dataset.csv`](../outputs/final_features/physiology_mwl_analysis_dataset.csv) as the authoritative, read-only analysis input. They do not rebuild or modify preprocessing results.

## Notebook order

1. `01_dataset_overview.ipynb` — validates dimensions, coverage, missingness, block composition, and numerical sanity.
2. `02_mwl_exploration.ipynb` — explores MWL distributions, progression, participant trajectories, and descriptive within-/between-participant variability. It does not model physiological associations.
3. `03_physiological_features.ipynb` — reserved for physiological feature exploration.
4. `04_physiology_mwl_relationship.ipynb` — reserved for physiology–MWL relationships.
5. `05_mixed_effects_models.ipynb` — reserved for mixed-effects modeling.

Run notebooks from either the repository root or this directory. Each notebook locates the repository root automatically with `pathlib`.

## Outputs

- Figures: `postprocessing/outputs/figures/`
- Reporting tables: `postprocessing/outputs/tables/`

The first two notebooks expose configuration variables near the top. In particular, `02_mwl_exploration.ipynb` can include or exclude the single approved imputed MWL rating with `INCLUDE_IMPUTED_MWL`.

Install the notebook environment with:

```bash
python -m pip install -r requirements.txt
jupyter lab
```

Postprocessing outputs are additional analysis artifacts and never replace preprocessing data, QC metadata, or final physiological features.
