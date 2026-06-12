# Gaming vs Academic — repo guide

Data Engineering project (**UFRO**). Final project goal: estimate a student's grade (`grades`,
0–100 scale) from their gaming, study, and rest routines — regression, eventually a neural
network. The current deliverable is an **association-rules problem** (Apriori) as an interpretable
step before modeling.

Team: Cristóbal Cariqueo, Alvaro Parra, Jun Sáez, Felipe Sequel.

Note: notebooks and docs are written in Spanish; this guide is in English.

## Layout

| Path | What it is |
|---|---|
| `data/raw/Gaming_Academic_Performance.csv` | Original dataset (Kaggle, 8000 students). **Untouched, do not edit** — it's the backup. |
| `data/processed/dataset_discretizado.csv` | Output of notebook 01 (7865 × 13). Generated, don't hand-edit. |
| `data/processed/reglas_todas.csv` | Output of notebook 03: all 7886 rules (no lift filter). |
| `notebooks/miercoles10/00_analisis_completo.ipynb` | Full self-contained pipeline (raw → rules). |
| `notebooks/miercoles10/01_exploracion_discretizacion.ipynb` | Profiling, outlier cleaning, quartile discretization. Generates `data/processed/`. |
| `notebooks/miercoles10/02_reglas_asociacion.ipynb` | Problem definition, Apriori, rules, conclusions. |
| `notebooks/viernes12/03_export_reglas.ipynb` | Reproduces the notebook-02 mining and exports all 7886 rules (no lift filter) to CSV. |

The `notebooks/` subfolders (`miercoles10/`, `viernes12/`) are team work sessions; they do not
imply separate pipelines.

## Data pipeline

Cleaning: drop 135 out-of-scale `grades` rows (134 with `> 100`, 1 with `== 0`) → 7865 rows.
Discretization: 10 numeric columns to quartiles Q1–Q4 (`pd.qcut`); 3 categoricals (`gender`,
`gaming_genre`, `stress_level`) kept as-is. Apriori over one-hot: `min_support=0.05, max_len=4`
→ 1767 frequent itemsets. From those: **7886 rules** total (no lift filter), of which **4808**
have lift ≥ 1.1 (the cutoff notebook 02 reports). Those counts are reference figures: if they
change, something in the data or parameters changed (notebook 03 pins both with `assert`s).

## Conventions

- **Path resolution in notebooks:** each notebook walks up from `Path.cwd()` until it finds the
  directory containing `data/`. Works whether the kernel starts at the repo root or in the
  notebook's own folder. When creating new notebooks, copy that block — don't hardcode relative
  paths.
- **All analysis evidence lives in versioned artifacts** (`.ipynb` notebooks or scripts in the
  repo), executed with outputs saved. No `python -c` against the dataset for conclusions: if it
  runs against the data, it goes in a repo file. One-liners only for environment checks.
- Comments only where non-obvious. Spanish in notebooks and docs.

## Environment

```bash
.venv/bin/jupyter lab          # venv already created
.venv/bin/jupyter nbconvert --to notebook --execute --inplace <path.ipynb>   # re-run
```

CachyOS / Arch; system packages via pacman/AUR. Python deps in `requirements.txt`
(pandas, mlxtend, matplotlib, seaborn, jupyter).
