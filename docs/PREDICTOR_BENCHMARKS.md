# Predictor Benchmarks

The cleaned pipeline provides a common interface for:

- `extra_trees`
- `svr`
- `random_forest`
- `xgboost` if the optional dependency is installed

All predictors use the same deterministic split, feature matrix, and target column from the YAML config.

## Features

The default feature matrix concatenates:

1. Numeric `feature_columns`, such as `params` and `flops`.
2. Expanded numeric architecture encoding from `chromosome`.

## Metrics

For each split, metrics are saved to `metrics/predictor_metrics.csv`:

- MAE, lower is better.
- RMSE, lower is better.
- R2, higher is better.
- Pearson correlation, higher is better.
- Spearman correlation, higher is better.

## Commands

```bash
python scripts/02_train_predictors.py --config configs/toy_example.yaml --model extra_trees
python scripts/02_train_predictors.py --config configs/toy_example.yaml --model svr
```

For XGBoost:

```bash
pip install -r requirements-optional.txt
python scripts/02_train_predictors.py --config configs/predictor_benchmark_template.yaml --model xgboost
```
