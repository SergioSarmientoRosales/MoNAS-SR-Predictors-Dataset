from __future__ import annotations

import argparse

from _bootstrap import add_src_to_path

add_src_to_path()

from monas_sr_predictors.config import load_config
from monas_sr_predictors.pipeline import validate_dataset_stage


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the encoded SR architecture dataset.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    frame = validate_dataset_stage(config)
    print(f"Validated {len(frame)} rows. Report: {config.run_dir / 'dataset_validation.json'}")


if __name__ == "__main__":
    main()
