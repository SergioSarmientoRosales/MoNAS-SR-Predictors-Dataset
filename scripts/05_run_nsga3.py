from __future__ import annotations

import argparse

from _bootstrap import add_src_to_path

add_src_to_path()

from monas_sr_predictors.config import load_config
from monas_sr_predictors.pipeline import nsga3_stage


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a lightweight NSGA-III-style selection stage.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    selected = nsga3_stage(config)
    print(f"Wrote {len(selected)} selected architectures to {config.run_dir / 'pareto_fronts' / 'nsga3_selection.csv'}")


if __name__ == "__main__":
    main()
