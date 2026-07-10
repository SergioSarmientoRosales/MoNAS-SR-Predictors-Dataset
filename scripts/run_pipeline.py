from __future__ import annotations

import argparse

from _bootstrap import add_src_to_path

add_src_to_path()

from monas_sr_predictors.config import load_config
from monas_sr_predictors.pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full clean MoNAS SR predictors pipeline.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    run_pipeline(config)
    print(f"Pipeline complete. Outputs are under {config.run_dir}")


if __name__ == "__main__":
    main()
