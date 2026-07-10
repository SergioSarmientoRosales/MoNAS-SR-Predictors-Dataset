from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from monas_sr_predictors.config import load_config
from monas_sr_predictors.pipeline import run_pipeline


def test_pipeline_runs(tmp_path: Path) -> None:
    config = load_config(Path("configs/toy_example.yaml"))
    config = type(config)(**{**config.__dict__, "run_name": "pytest_toy", "output_dir": tmp_path, "plot": False})
    run_pipeline(config)
    assert (config.run_dir / "dataset_validation.json").exists()
    assert (config.run_dir / "metrics" / "predictor_metrics.csv").exists()
    assert (config.run_dir / "metrics" / "zero_cost_metrics.csv").exists()
    assert (config.run_dir / "pareto_fronts" / "nsga3_selection.csv").exists()
    assert not (config.run_dir / "figures" / "predicted_vs_true.png").exists()
    assert not (config.run_dir / "figures" / "nsga3_selection.png").exists()


def test_minimal_dataset_example_runs(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, str(repo_root / "examples" / "minimal_dataset_example.py")],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "Validation report" in result.stdout
