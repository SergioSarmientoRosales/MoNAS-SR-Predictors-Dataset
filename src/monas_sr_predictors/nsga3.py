from __future__ import annotations

import numpy as np
import pandas as pd

from monas_sr_predictors.config import PipelineConfig
from monas_sr_predictors.dataset import objective_matrix
from monas_sr_predictors.pareto import nondominated_frame


def reference_directions(n_objectives: int, n_points: int) -> np.ndarray:
    """Create deterministic reference directions for small examples."""
    if n_objectives == 2:
        grid = np.linspace(0.0, 1.0, max(2, n_points))
        return np.column_stack([grid, 1.0 - grid])
    rng = np.random.default_rng(0)
    raw = rng.random((n_points, n_objectives))
    return raw / raw.sum(axis=1, keepdims=True)


def nsga3_style_selection(frame: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    """Small deterministic NSGA-III-style environmental selection.

    This is a lightweight reproducibility path for examples. The heavy historical
    NSGA-III/TensorFlow search remains in `Codes/`.
    """
    objectives = objective_matrix(frame, config.objectives)
    front = nondominated_frame(frame, objectives)
    if len(front) <= config.nsga3_archive_size:
        return front.reset_index(drop=True)

    front_objectives = objective_matrix(front, config.objectives)
    mins = front_objectives.min(axis=0)
    spans = np.where(front_objectives.max(axis=0) - mins == 0, 1.0, front_objectives.max(axis=0) - mins)
    normalized = (front_objectives - mins) / spans
    directions = reference_directions(normalized.shape[1], config.nsga3_archive_size)

    selected: list[int] = []
    for direction in directions:
        direction = direction / (np.linalg.norm(direction) or 1.0)
        distances = []
        for point in normalized:
            projection = np.dot(point, direction) * direction
            distances.append(np.linalg.norm(point - projection))
        order = np.argsort(distances, kind="mergesort")
        for index in order:
            if int(index) not in selected:
                selected.append(int(index))
                break
        if len(selected) >= config.nsga3_archive_size:
            break

    while len(selected) < config.nsga3_archive_size:
        objective_sum = normalized.sum(axis=1)
        for index in np.argsort(objective_sum, kind="mergesort"):
            if int(index) not in selected:
                selected.append(int(index))
                break

    return front.iloc[selected].reset_index(drop=True)
