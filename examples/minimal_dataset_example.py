from __future__ import annotations

import argparse
from pathlib import Path

from _example_bootstrap import REPO_ROOT

from monas_sr_predictors.config import load_config
from monas_sr_predictors.pipeline import validate_dataset_stage


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and summarize the toy architecture dataset.")
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "toy_example.yaml"))
    args = parser.parse_args()
    config = load_config(args.config)
    config = type(config)(**{**config.__dict__, "run_name": "examples/minimal_dataset"})
    frame = validate_dataset_stage(config)
    print(frame.describe(include="all").to_string())
    print(f"Validation report: {config.run_dir / 'dataset_validation.json'}")


if __name__ == "__main__":
    main()
