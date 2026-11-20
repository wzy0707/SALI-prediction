# Early prediction of sepsis-associated liver injury in elderly ICU patients

Reproducible analysis code for the manuscript:

**Machine learning-based early prediction of sepsis-associated liver injury in elderly ICU patients: a retrospective cohort study based on MIMIC-IV with external validation in eICU-CRD**

## Overview

The primary deployable model is a **five-variable interpretable logistic score** (baseline total bilirubin, partial thromboplastin time, sodium, platelets, lactate). The six machine-learning models (logistic regression, XGBoost, LightGBM, MLP, support-vector machine, CatBoost) are reported as **internal performance benchmarks**. The external validation in eICU-CRD is performed on the five-variable score because the full 70-feature machine-learning models cannot be aligned across the two databases.

The repository covers cohort extraction, leakage-free preprocessing, feature selection, model training, internal evaluation, external validation, robustness analyses, and publication-quality vector figure generation.

## Data availability

No patient-level data are included. MIMIC-IV and eICU-CRD require credentialed access under the PhysioNet data use agreement and cannot be redistributed.

- **MIMIC-IV v3.1:** https://physionet.org/content/mimiciv/
- **eICU-CRD:** https://physionet.org/content/eicu-crd/

Qualified researchers obtain access after completing PhysioNet credentialing (CITI Data or Specimens Only Research training).

## Repository layout

### `sql/` — cohort construction

| File | Role | Description |
| --- | --- | --- |
| `SALI_一键取数_最终建模数据.sql` | Extraction | MIMIC-IV cohort construction and the final modeling dataset |
| `eicu_external_validation.sql` | Extraction | eICU-CRD external validation cohort |

### `code/` — preprocessing, training and internal evaluation

| File | Role | Description |
| --- | --- | --- |
| `S0_干净重跑_预处理.R` | Preprocessing | Seed-42 stratified 7:3 split and leakage-free imputation; writes the train/test files for the Python pipeline |
| `R old liver.R` | Preprocessing | R implementation of the same leakage-free split, preprocessing and feature selection |
| `python old liver.py` | Training / evaluation | Six-model nested cross-validation, five-variable logistic score, SHAP, calibration and decision-curve analysis |
| `B2_internal_ppv.py` | Evaluation | PPV and full confusion matrices for the six models and the five-variable score on the internal test set |
| `B4_hyperparameter_table.py` | Evaluation | Hyperparameter search grids, final configurations and random seeds |
| `B8_auc_difference_ci.py` | Evaluation | Paired bootstrap confidence interval for the CatBoost vs five-variable ROC-AUC difference |
| `B6_figure_redraw.R` | Figures | Redrawn Boruta, calibration and nomogram vector figures |
| `SALI_全部图片_R矢量PDF.R` | Figures | R split, LASSO, Boruta, nomogram, calibration and vector PDF export |
| `SALI_模型评估_Python矢量PDF.py` | Figures | Python model-evaluation vector PDFs (ROC/PR, SHAP, decision curve, calibration) |

### `external_validation/` — eICU-CRD validation

| File | Role | Description |
| --- | --- | --- |
| `eicu_external_validation.py` | Validation | Frozen-model external validation, calibration and recalibration |
| `eicu_alignment_check.py` | Validation | Cross-cohort variable alignment and data-quality checks |
| `B1_eicu_cluster_bootstrap.py` | Validation | Hospital-level cluster bootstrap confidence intervals and confusion matrix |
| `B3_eicu_baseline_smd.py` | Validation | Baseline characteristics table and standardized mean differences between cohorts |
| `eicu_validation_figures.R` | Figures | External ROC and calibration figures |
| `B10_Fig4_vector.R` | Figures | True-vector versions of Figure 4 (external ROC and calibration) |

### `robustness/` — sensitivity and robustness analyses

| File | Role | Description |
| --- | --- | --- |
| `competing_risk.R` | Competing risk | Cumulative incidence and Fine–Gray model with death as competing event |
| `B9_FigA8_CIF.R` | Competing risk | Vector cumulative-incidence figure for SALI and competing death over 7 days |
| `ipw_nofollow.py` | Missing follow-up | Inverse-probability weighting for the no-follow-up subset |
| `B5_composite_censoring_subgroup.py` | Sensitivity | Composite outcome, censoring bounds and subgroup performance |
| `B7_ptt_replacement.py` | Sensitivity | PT/INR replacement analysis supporting retention of PTT |
| `sensitivity_analysis.py` | Sensitivity | Additional sensitivity analyses and model comparisons |
| `persistent_rerun.R` | Sensitivity | Persistent-event reanalysis |

## Software requirements

**Python** (tested on 3.12; also runs on 3.10). Install the pinned environment with:

```
python -m pip install -r requirements.txt
```

Key packages: pandas 2.2.3, numpy 2.5.4, scikit-learn 1.9.1, scipy 1.17.1, xgboost 3.4.1, lightgbm 4.7.0, catboost 1.2.10, shap 0.54.1, matplotlib 3.11.0, seaborn 0.13.2.

**R 4.5.1** with `caret`, `dplyr`, `ggplot2`, `glmnet`, `rms`, `Hmisc`, `Boruta`, `naniar`, `survival`, `cmprsk`. Run `export_session_info.R` once to record the exact R and package versions into `R_session_info.txt`.

## Reproducibility

1. Use the `sql/` scripts to build the MIMIC-IV and eICU-CRD cohorts from PhysioNet.
2. Source `code/S0_干净重跑_预处理.R` to create the leakage-free split and imputed files.
3. Run `code/python old liver.py` to train the six models and the five-variable score and produce internal evaluations.
4. Run the scripts in `external_validation/` for the eICU-CRD validation.
5. Run the scripts in `robustness/` for the competing-risk, sensitivity and subgroup analyses.

All random seeds are fixed at 42. Scaling, one-hot encoding and imputation are fitted on the training data only and applied to the test data, preventing information leakage.

## Deployment status

This is a retrospectively developed and validated research tool. It is **not** offered as a web service or software application, has not been adopted in clinical practice, and is not intended for standalone diagnosis. The current technology readiness level is TRL 4. Prospective evaluation and local calibration are required before any clinical deployment; the intended use is early rule-out monitoring and triage at the first-24-hour time point.

## License

The code is released under the MIT License (see `LICENSE`).
