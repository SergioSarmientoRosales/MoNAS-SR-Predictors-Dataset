from __future__ import annotations

import argparse

from _bootstrap import add_src_to_path

add_src_to_path()

from monas_sr_predictors.config import load_config
from monas_sr_predictors.pipeline import train_predictors_stage


def main() -> None:
    parser = argparse.ArgumentParser(description="Train configured predictor models.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", help="Train only one model, e.g. svr, extra_trees, xgboost.")
    args = parser.parse_args()
    config = load_config(args.config)
    metrics, _ = train_predictors_stage(config, model=args.model)
    print(f"Wrote {len(metrics)} predictor metric rows to {config.run_dir / 'metrics' / 'predictor_metrics.csv'}")


if __name__ == "__main__":
    main()
