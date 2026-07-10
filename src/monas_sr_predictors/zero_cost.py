from __future__ import annotations

import numpy as np
import pandas as pd

from monas_sr_predictors.config import PipelineConfig
from monas_sr_predictors.dataset import parse_architecture_encoding
from monas_sr_predictors.metrics import regression_metrics


def synflow_like_score(encoding: list[float]) -> float:
    """Deterministic toy proxy used for examples, not a replacement for true SynFlow.

    The historical repository includes SynFlow seed outputs. This function gives a
    lightweight, architecture-dependent score so CI and examples can exercise the
    zero-cost proxy interface without building TensorFlow models.
    """
    values = np.asarray(encoding, dtype=float)
    weights = np.linspace(1.0, 2.0, len(values))
    return float(np.sum((values + 1.0) * weights))


def compute_zero_cost(frame: pd.DataFrame, config: PipelineConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = frame[[config.architecture_id_column, config.target_columns[0], "split"]].copy()
    for proxy in config.zero_cost_proxies:
        if proxy.lower() != "synflow_like":
            raise ValueError(f"Unsupported zero-cost proxy in clean pipeline: {proxy}")
        rows[proxy] = [
            synflow_like_score(parse_architecture_encoding(value))
            for value in frame[config.architecture_column]
        ]

    metrics_rows: list[dict[str, object]] = []
    target = config.target_columns[0]
    for proxy in config.zero_cost_proxies:
        for split_name in ["train", "validation", "test"]:
            subset = rows[rows["split"] == split_name]
            if subset.empty:
                continue
            values = regression_metrics(subset[target].to_numpy(dtype=float), subset[proxy].to_numpy(dtype=float))
            for metric, value in values.items():
                metrics_rows.append(
                    {"proxy": proxy, "split": split_name, "target": target, "metric": metric, "value": value}
                )
    return rows, pd.DataFrame(metrics_rows)
