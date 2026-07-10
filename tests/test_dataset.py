from __future__ import annotations

from pathlib import Path

from monas_sr_predictors.config import load_config
from monas_sr_predictors.dataset import (
    deterministic_split,
    load_dataset,
    parse_architecture_encoding,
    validate_dataset,
)


def test_parse_architecture_encoding() -> None:
    assert parse_architecture_encoding("[1, 2, 3]") == [1.0, 2.0, 3.0]


def test_dataset_validation_and_split_are_deterministic() -> None:
    config = load_config(Path("configs/toy_example.yaml"))
    frame = load_dataset(config)
    report = validate_dataset(frame, config)
    assert report["rows"] == 12
    assert report["architecture_encoding_length"] == 8
    first = deterministic_split(frame, config)["split"].tolist()
    second = deterministic_split(frame, config)["split"].tolist()
    assert first == second
    assert set(first) == {"train", "validation", "test"}
