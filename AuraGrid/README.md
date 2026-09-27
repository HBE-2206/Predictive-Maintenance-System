# AuraGrid Systems — Predictive Maintenance

Machine-learning project for failure-type classification and repair-cost prediction.

## Objectives
- Predict `Failure_Type` from production/energy/availability indicators.
- Predict `Repair_Cost_USD`.

## Features used
- `Factory_Line`
- `Operation_Type`
- `Total_Energy_kWh`
- `Production_Units`
- `Machine_Availability_Pct`

## Models
### Classification
- Logistic Regression — baseline
- Random Forest — advanced

### Regression
- Linear Regression — baseline
- Random Forest — advanced

## Evaluation
Classification: Accuracy, Weighted F1, Macro F1, Confusion Matrix, Classification Report.

Regression: RMSE, MAE, R².

## Data note
The supplied dataset contains 150 records. `Failure_Type` contains missing target values, so rows without a failure label are excluded from supervised classification instead of assigning an invented label.

## Observed hold-out results
| Task | Baseline | Advanced |
|---|---:|---:|
| Classification Accuracy | 33.33% | 11.11% |
| Classification Weighted F1 | 0.2708 | 0.0741 |
| Repair Cost RMSE | $430.00 | $441.41 |
| Repair Cost MAE | $388.99 | $396.13 |
| Repair Cost R² | -0.0808 | -0.1390 |

These results are a baseline benchmark, not production validation. The small dataset and limited requested feature set provide limited predictive signal.

## Recommended improvements
1. Define an explicit `No Failure` class if appropriate.
2. Experiment with `Defects_Count`, `Defect_Type`, and `Maintenance_Events` after checking for leakage.
3. Add historical/rolling operational features.
4. Increase dataset size.
5. Use cross-validation and hyperparameter tuning on a larger dataset.
