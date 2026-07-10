from __future__ import annotations

import argparse

from _example_bootstrap import REPO_ROOT

from monas_sr_predictors.config import load_config
from monas_sr_predictors.pipeline import nsga3_stage, split_dataset_stage, validate_dataset_stage


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a tiny NSGA-III-style selection on toy data.")
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "toy_example.yaml"))
    args = parser.parse_args()
    config = load_config(args.config)
    config = type(config)(**{**config.__dict__, "run_name": "examples/nsga3"})
    validate_dataset_stage(config)
    split_dataset_stage(config)
    selected = nsga3_stage(config)
    print(selected[[config.architecture_id_column, "valid_psnr", "params", "flops"]].to_string(index=False))


if __name__ == "__main__":
    main()
