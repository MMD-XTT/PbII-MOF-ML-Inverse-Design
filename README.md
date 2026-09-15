# PbII-MOF-ML-Inverse-Design

Reproducibility package for the data-driven screening and inverse design of metal-organic frameworks (MOFs) for Pb(II) adsorption. The workflow integrates GCMC simulation, machine-learning prediction, SHAP interpretation, paired statistical comparison, and tangent-adaptive genetic algorithm (TAGA) optimization.

## Workflow

1. Generate Pb(II)-adsorption labels for 2,000 stratified MOFs by GCMC simulation.
2. Compare LR, SVM, RF, XGBoost, Extra Trees, and CatBoost models using five-fold cross-validation.
3. Use the selected CatBoost model for full-database Pb(II) prediction and ranking.
4. Interpret feature contributions through SHAP global-importance and beeswarm analyses.
5. Construct a 3,854-MOF genotype-operable library and fit an RF TAGA fitness model.
6. Apply TAGA to generate and rank new MOF configurations, then retain the generation-level outputs and candidate list.

## Repository Layout

```text
data/       Essential datasets, cross-validation outputs, and TAGA generation records
model/      Final CatBoost Pb(II)-prediction model and RF TAGA fitness model
scripts/    Compact scripts for training, statistics, SHAP, and TAGA optimization
```

## Included Data

- 2,000 GCMC-labeled MOFs for predictive-model development.
- Full-database Pb(II) values combining GCMC labels and CatBoost predictions.
- Curated 3,854-MOF genotype-operable GA library.
- Five-fold RF fitness-model results and out-of-fold predictions.
- TAGA generation summaries, complete population records, and top 50 generated candidates.

## Core Scripts

- `train_six_models.py`: LR, SVM, RF, XGBoost, Extra Trees, and CatBoost comparison.
- `train_catboost_pbii_model.py`: CatBoost Pb(II)-prediction training.
- `train_rf_taga_fitness_model.py`: six-locus RF TAGA fitness-model training.
- `wilcoxon_signed_rank_test.py`: paired Wilcoxon signed-rank tests with Holm adjustment.
- `cohens_d_effect_size.py`: paired Cohen's *d* calculations.
- `shap_bar_plot.py` and `shap_beeswarm_plot.py`: model-interpretation figures.
- `run_taga_optimization.py`: original TAGA implementation retained without modification.

## Installation and Use

```powershell
pip install -r requirements.txt
python scripts/train_catboost_pbii_model.py --input <prepared_table> --target <pbii_loading> --features <feature_1,feature_2,...> --model-out model/CatBoost_PbII_Prediction_Model.joblib
```

The generic scripts use command-line arguments rather than fixed input locations. Supply the relevant local table and column names before execution. The final fitted models are retained in `model/` for direct reproducibility.
