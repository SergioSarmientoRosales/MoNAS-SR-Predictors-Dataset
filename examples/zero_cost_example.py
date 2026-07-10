from __future__ import annotations

import argparse

from _example_bootstrap import REPO_ROOT

from monas_sr_predictors.config import load_config
from monas_sr_predictors.pipeline import split_dataset_stage, validate_dataset_stage, zero_cost_stage


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute the lightweight zero-cost proxy on toy data.")
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "toy_example.yaml"))
    args = parser.parse_args()
    config = load_config(args.config)
    config = type(config)(**{**config.__dict__, "run_name": "examples/zero_cost"})
    validate_dataset_stage(config)
    split_dataset_stage(config)
    scores, metrics = zero_cost_stage(config)
    print(scores.head().to_string(index=False))
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
