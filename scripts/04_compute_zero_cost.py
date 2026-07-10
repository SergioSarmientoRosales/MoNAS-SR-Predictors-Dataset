from __future__ import annotations

import argparse

from _bootstrap import add_src_to_path

add_src_to_path()

from monas_sr_predictors.config import load_config
from monas_sr_predictors.pipeline import zero_cost_stage


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute/load zero-cost proxy scores.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    _, metrics = zero_cost_stage(config)
    print(f"Wrote {len(metrics)} zero-cost metric rows to {config.run_dir / 'metrics' / 'zero_cost_metrics.csv'}")


if __name__ == "__main__":
    main()
