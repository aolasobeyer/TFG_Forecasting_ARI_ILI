# TFG_Forecasting_ARI_ILI
# ARI and ILI incidence forecasting

This repository contains the code used for my Bachelor Thesis on short-term forecasting of weekly ARI and ILI incidence across European countries.

The project compares several forecasting approaches, including statistical models, machine learning models, deep learning models, forecast combinations, Diebold-Mariano tests and an external comparison with RespiCast forecasts.

## Repository structure

```text
notebooks/   Main notebooks used in the project
src/         Helper Python functions used by the notebooks
figures/     Final figures used in the report
````

The raw data and some generated result files are expected to follow the same local folder structure used during the project.

## Notebook order

The notebooks are numbered according to the main workflow:

1. `01_data_loading.ipynb`
   Loads the ARI and ILI incidence files.

2. `01b_raw_data_duplicate_audit.ipynb`
   Checks duplicated country-week observations in the raw files.

3. `02_time_index_and_gaps.ipynb`
   Builds the weekly time index and analyses missing values and gaps.

4. `03_imputation.ipynb`
   Applies the training imputation rules used in the project.

5. `04_baselines_rolling_eval.ipynb`
   Evaluates the simple baseline forecasting methods.

6. `05_sarima_rolling_eval.ipynb`
   Evaluates SARIMA and autoARIMA models.

7. `06_random_forest_rolling_eval.ipynb`
   Evaluates local random forest models.

8. `07_random_forest_all_infections_all_countries.ipynb`
   Evaluates the pooled random forest model trained across diseases and countries.

9. `08_deep_learning.ipynb`
   Evaluates the deep learning models.

10. `09_bivariate_varima_rolling_eval.ipynb.ipynb`
    Evaluates the bivariate VARIMA benchmark.

11. `10_summary_tables.ipynb`
    Builds summary tables from the validation results.

12. `11_final_test_2024_2025_winners_eval.ipynb`
    Runs the final 2024-2025 test evaluation for the selected models and ensembles.

13. `12_diebold_mariano_final_test.ipynb`
    Applies the Diebold-Mariano tests to the final test results.

14. `13_respicast_comparison_log2.ipynb`
    Earlier RespiCast comparison using a log2-based approach. This is kept for traceability.

15. `13b_respicast_comparison_log2_tables.ipynb`
    Tables related to the earlier log2-based RespiCast comparison.

16. `14_respicast_comparison_mae_scaled.ipynb`
    Final RespiCast comparison used in the report, based on MAE and mean scaled absolute error.

17. `15_audit_respicast_comparison.ipynb`
    Additional checks for the RespiCast comparison.

18. `16_report_assets_tables_figures.ipynb`
    Generates the final tables and figures used in the report.

## Notes

The final report uses the MAE and mean scaled absolute error RespiCast comparison from notebook `14_respicast_comparison_mae_scaled.ipynb`.

The log2-based RespiCast notebooks are not the final comparison used in the report, but they are kept to show the previous exploratory step.

The notebooks were developed for an academic project and may require the same local data and results folders used during the thesis.
