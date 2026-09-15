# Data

The essential analysis tables are included with descriptive names:

- `GCMC_2000_Training_Dataset.xlsx`: 2,000 stratified GCMC-labeled MOFs for Pb(II) model development.
- `Full_Database_GCMC_and_CatBoost_Predictions.xlsx`: full-database Pb(II) ranking combining GCMC labels and CatBoost predictions.
- `GA_Curated_Genotype_Library_3854.xlsx`: curated 3,854-MOF genotype-operable library.
- `TAGA_RF_FiveFold_CV_Performance.xlsx` and `TAGA_RF_FiveFold_OOF_Predictions.xlsx`: RF fitness-model validation outputs.
- `TAGA_Generation_Summary.xlsx`: generation-level fitness trajectory.
- `TAGA_All_Generations_Population.xlsx`: every TAGA generation and individual.
- `TAGA_Top50_Generated_Candidates.xlsx`: final high-ranked generated candidates.

Scripts accept the table and field names through command-line arguments, so no input location is hard-coded.
