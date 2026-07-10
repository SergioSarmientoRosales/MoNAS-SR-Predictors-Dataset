from __future__ import annotations

import argparse

from _example_bootstrap import REPO_ROOT

from monas_sr_predictors.config import load_config
from monas_sr_predictors.pipeline import split_dataset_stage, train_predictors_stage, validate_dataset_stage


def main() -> None:
    parser = argparse.ArgumentParser(description="Train lightweight predictors on the toy dataset.")
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "toy_example.yaml"))
    parser.add_argument("--model", default="extra_trees")
    args = parser.parse_args()
    config = load_config(args.config)
    config = type(config)(**{**config.__dict__, "run_name": "examples/train_predictor"})
    validate_dataset_stage(config)
    split_dataset_stage(config)
    metrics, _ = train_predictors_stage(config, model=args.model)
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
