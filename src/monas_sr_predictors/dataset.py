from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd

from monas_sr_predictors.config import ObjectiveSpec, PipelineConfig


def parse_architecture_encoding(value: object) -> list[float]:
    """Parse a string/list architecture encoding into numeric features."""
    if isinstance(value, list):
        parsed = value
    else:
        try:
            parsed = ast.literal_eval(str(value))
        except (SyntaxError, ValueError) as exc:
            raise ValueError(f"Invalid architecture encoding: {value!r}") from exc
    if not isinstance(parsed, list):
        raise ValueError(f"Architecture encoding must be a list: {value!r}")
    try:
        return [float(item) for item in parsed]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Architecture encoding contains non-numeric values: {value!r}") from exc


def load_dataset(config: PipelineConfig) -> pd.DataFrame:
    """Load a configured dataset CSV and normalize obvious column types."""
    header = 0 if config.csv_has_header else None
    frame = pd.read_csv(config.input_path, header=header)
    if not config.csv_has_header:
        if not config.csv_column_names:
            raise ValueError("csv_column_names is required when csv_has_header is false.")
        frame.columns = list(config.csv_column_names)

    if config.architecture_id_column not in frame.columns:
        frame.insert(0, config.architecture_id_column, [f"arch_{index:06d}" for index in range(len(frame))])
    frame[config.architecture_id_column] = frame[config.architecture_id_column].astype(str)

    return frame.reset_index(drop=True)


def validate_dataset(frame: pd.DataFrame, config: PipelineConfig) -> dict[str, object]:
    """Validate required columns, numeric targets/objectives, and encodings."""
    required = {config.architecture_id_column, config.architecture_column}
    required.update(config.target_columns)
    required.update(objective.column for objective in config.objectives)
    required.update(config.feature_columns)

    missing = sorted(column for column in required if column not in frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    numeric_columns = set(config.target_columns)
    numeric_columns.update(objective.column for objective in config.objectives)
    numeric_columns.update(config.feature_columns)
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    invalid_numeric = {
        column: int(frame[column].isna().sum())
        for column in numeric_columns
        if frame[column].isna().any()
    }
    if invalid_numeric:
        raise ValueError(f"Numeric columns contain missing/non-numeric values: {invalid_numeric}")

    parsed_lengths = [len(parse_architecture_encoding(value)) for value in frame[config.architecture_column]]
    if len(set(parsed_lengths)) != 1:
        raise ValueError(f"Architecture encodings have inconsistent lengths: {sorted(set(parsed_lengths))}")

    duplicate_count = int(frame[config.architecture_id_column].duplicated().sum())
    return {
        "rows": int(len(frame)),
        "columns": list(frame.columns),
        "architecture_encoding_length": int(parsed_lengths[0]),
        "duplicate_architecture_ids": duplicate_count,
        "targets": list(config.target_columns),
        "objectives": [
            {"column": objective.column, "direction": objective.direction}
            for objective in config.objectives
        ],
    }


def write_validation_report(report: dict[str, object], path: str | Path) -> None:
    Path(path).write_text(json.dumps(report, indent=2), encoding="utf-8")


def feature_matrix(frame: pd.DataFrame, config: PipelineConfig) -> tuple[np.ndarray, list[str]]:
    """Create predictor features from configured feature columns and encoding."""
    matrices: list[np.ndarray] = []
    names: list[str] = []

    if config.feature_columns:
        matrices.append(frame[list(config.feature_columns)].to_numpy(dtype=float))
        names.extend(config.feature_columns)

    encodings = [parse_architecture_encoding(value) for value in frame[config.architecture_column]]
    encoding_matrix = np.asarray(encodings, dtype=float)
    matrices.append(encoding_matrix)
    names.extend([f"encoding_{index}" for index in range(encoding_matrix.shape[1])])

    return np.column_stack(matrices), names


def deterministic_split(frame: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    """Assign train/validation/test splits with a fixed seed."""
    rng = np.random.default_rng(config.seed)
    indices = np.arange(len(frame))
    rng.shuffle(indices)

    n_train = int(round(len(frame) * config.split.train))
    n_validation = int(round(len(frame) * config.split.validation))
    split_labels = np.empty(len(frame), dtype=object)
    split_labels[indices[:n_train]] = "train"
    split_labels[indices[n_train : n_train + n_validation]] = "validation"
    split_labels[indices[n_train + n_validation :]] = "test"

    out = frame.copy()
    out["split"] = split_labels
    return out


def objective_matrix(frame: pd.DataFrame, objectives: tuple[ObjectiveSpec, ...]) -> np.ndarray:
    """Return objectives as minimization values for Pareto calculations."""
    columns = []
    for objective in objectives:
        values = frame[objective.column].to_numpy(dtype=float)
        columns.append(-values if objective.direction == "maximize" else values)
    return np.column_stack(columns)
