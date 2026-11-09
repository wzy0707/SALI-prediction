# Early prediction of sepsis-associated liver injury in elderly ICU patients

Reproducible analysis code for the manuscript:

**Machine learning-based early prediction of sepsis-associated liver injury in elderly ICU patients: MIMIC-IV with external validation in eICU-CRD**

## Overview

This repository contains the code used to build and validate the machine-learning prediction model. It performs cohort construction, feature selection, model training, external validation, and vector figure generation.

## Data availability

The underlying datasets are not included because they require credentialed access under the PhysioNet data use agreement.

- **MIMIC-IV v3.1:** https://physionet.org/content/mimiciv/
- **eICU-CRD:** https://physionet.org/content/eicu-crd/

Qualified researchers can access the data after completing the PhysioNet credentialing process (CITI Data or Specimens Only Research training).

## Repository contents

| File | Description |
| --- | --- |
| `code/SALI_全部图片_R矢量PDF.R` | R side: split, leakage-free imputation, LASSO, Boruta, nomogram, calibration, competing-risk CIF, external-validation figures; exports vector PDFs |
| `code/SALI_模型评估_Python矢量PDF.py` | Python side: six-model training, ROC/PR, SHAP, decision curve, CatBoost calibration; exports vector PDFs |
| `sql/SALI_一键取数_最终建模数据.sql` | SQL for MIMIC-IV cohort construction and the final modeling dataset |
| `sql/eicu_external_validation.sql` | SQL for the eICU-CRD external validation cohort |
| `external_validation/eicu_external_validation.py` | External validation, calibration and recalibration |
| `external_validation/eicu_alignment_check.py` | Cross-cohort variable alignment and data-quality checks |
| `external_validation/eicu_validation_figures.R` | External-validation ROC and calibration figures |
| `robustness/competing_risk.R` | Competing-risk cumulative incidence and Fine–Gray model |
| `robustness/ipw_nofollow.py` | Inverse-probability weighting for the no-follow-up subset |
| `robustness/sensitivity_analysis.py` | Sensitivity analyses and model comparisons |
| `robustness/persistent_rerun.R` | Persistent-event reanalysis |

## Software requirements

- R 4.5.1 with `caret`, `dplyr`, `ggplot2`, `glmnet`, `rms`, `Boruta`, `naniar`, `survival`, `cmprsk`
- Python 3.10 with `pandas`, `numpy`, `scikit-learn`, `scipy`, `matplotlib`, `shap`, `xgboost`, `lightgbm`, `catboost`

## Reproducibility

1. Extract the source tables from PhysioNet and build the analysis dataset.
2. Source `SALI_全部图片_R矢量PDF.R`, which writes the split files and vector PDFs.
3. Run `SALI_模型评估_Python矢量PDF.py` to generate the model-evaluation vector PDFs.

All random seeds are fixed at 42.

## License

The code is released under the MIT License.
