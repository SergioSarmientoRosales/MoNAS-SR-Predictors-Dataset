from __future__ import annotations

import argparse

from _bootstrap import add_src_to_path

add_src_to_path()

from monas_sr_predictors.config import load_config
from monas_sr_predictors.pipeline import evaluate_predictors_stage


def main() -> None:
    parser = argparse.ArgumentParser(description="Load or compute predictor evaluation metrics.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    metrics = evaluate_predictors_stage(config)
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
