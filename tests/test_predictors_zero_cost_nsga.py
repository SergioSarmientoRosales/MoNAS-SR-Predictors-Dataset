from __future__ import annotations

from pathlib import Path

import numpy as np

from monas_sr_predictors.config import load_config
from monas_sr_predictors.dataset import deterministic_split, load_dataset, objective_matrix
from monas_sr_predictors.nsga3 import nsga3_style_selection
from monas_sr_predictors.pareto import dominates, nondominated_mask
from monas_sr_predictors.predictors import train_predictors
from monas_sr_predictors.zero_cost import compute_zero_cost


def _split_frame():
    config = load_config(Path("configs/toy_example.yaml"))
    frame = deterministic_split(load_dataset(config), config)
    return config, frame


def test_pareto_dominance() -> None:
    assert dominates(np.array([1.0, 2.0]), np.array([2.0, 2.0]))
    assert not dominates(np.array([2.0, 1.0]), np.array([1.0, 2.0]))
    mask = nondominated_mask(np.array([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0], [0.8, 0.8]]))
    assert mask.tolist() == [True, True, True, False]


def test_predictor_training_on_toy_data() -> None:
    config, frame = _split_frame()
    config = type(config)(**{**config.__dict__, "predictors": ("extra_trees",)})
    metrics, predictions = train_predictors(frame, config)
    assert not metrics.empty
    assert not predictions.empty
    assert set(metrics["split"]) == {"train", "validation", "test"}


def test_zero_cost_interface() -> None:
    config, frame = _split_frame()
    scores, metrics = compute_zero_cost(frame, config)
    assert "synflow_like" in scores.columns
    assert not metrics.empty


def test_nsga3_selection_size() -> None:
    config, frame = _split_frame()
    selected = nsga3_style_selection(frame, config)
    assert 0 < len(selected) <= config.nsga3_archive_size
    objective_matrix(selected, config.objectives)
