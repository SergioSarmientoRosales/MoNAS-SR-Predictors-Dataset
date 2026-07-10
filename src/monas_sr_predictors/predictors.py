from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from monas_sr_predictors.config import PipelineConfig
from monas_sr_predictors.dataset import feature_matrix
from monas_sr_predictors.metrics import regression_metrics


def make_predictor(name: str, seed: int):
    """Create a predictor with deterministic defaults."""
    key = name.lower()
    if key in {"extra_trees", "extratrees", "et"}:
        return ExtraTreesRegressor(n_estimators=100, random_state=seed, n_jobs=1)
    if key in {"random_forest", "rf"}:
        return RandomForestRegressor(n_estimators=100, random_state=seed, n_jobs=1)
    if key == "svr":
        return Pipeline([("scale", StandardScaler()), ("model", SVR(C=10.0, epsilon=0.05))])
    if key in {"xgboost", "xgb"}:
        try:
            from xgboost import XGBRegressor
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "xgboost is optional. Install it with `pip install -r requirements-optional.txt`."
            ) from exc
        return XGBRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            subsample=1.0,
            colsample_bytree=1.0,
            random_state=seed,
            objective="reg:squarederror",
            n_jobs=1,
        )
    raise ValueError(f"Unknown predictor model: {name}")


def train_predictors(frame: pd.DataFrame, config: PipelineConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train configured predictors and return metrics/predictions."""
    x, feature_names = feature_matrix(frame, config)
    target = config.target_columns[0]
    y = frame[target].to_numpy(dtype=float)
    train_mask = frame["split"] == "train"

    metrics_rows: list[dict[str, object]] = []
    predictions_rows: list[pd.DataFrame] = []
    model_dir = config.run_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    for model_name in config.predictors:
        model = make_predictor(model_name, config.seed)
        model.fit(x[train_mask], y[train_mask])
        joblib.dump({"model": model, "feature_names": feature_names, "target": target}, model_dir / f"{model_name}.joblib")

        for split_name in ["train", "validation", "test"]:
            mask = frame["split"] == split_name
            if not mask.any():
                continue
            pred = model.predict(x[mask])
            values = regression_metrics(y[mask], pred)
            for metric, value in values.items():
                metrics_rows.append(
                    {"model": model_name, "split": split_name, "target": target, "metric": metric, "value": value}
                )
            split_predictions = frame.loc[mask, [config.architecture_id_column, target, "split"]].copy()
            split_predictions["model"] = model_name
            split_predictions["prediction"] = pred
            predictions_rows.append(split_predictions)

    metrics = pd.DataFrame(metrics_rows)
    predictions = pd.concat(predictions_rows, ignore_index=True) if predictions_rows else pd.DataFrame()
    return metrics, predictions
