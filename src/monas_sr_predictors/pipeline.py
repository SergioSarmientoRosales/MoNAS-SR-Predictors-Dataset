from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from monas_sr_predictors.config import PipelineConfig, dump_config, load_config
from monas_sr_predictors.dataset import (
    deterministic_split,
    load_dataset,
    validate_dataset,
    write_validation_report,
)
from monas_sr_predictors.nsga3 import nsga3_style_selection
from monas_sr_predictors.plotting import plot_pareto, plot_predicted_vs_true
from monas_sr_predictors.predictors import train_predictors
from monas_sr_predictors.zero_cost import compute_zero_cost


def ensure_run_dirs(config: PipelineConfig) -> None:
    for path in [
        config.run_dir,
        config.run_dir / "splits",
        config.run_dir / "models",
        config.run_dir / "predictions",
        config.run_dir / "metrics",
        config.run_dir / "pareto_fronts",
        config.run_dir / "figures",
        config.run_dir / "logs",
    ]:
        path.mkdir(parents=True, exist_ok=True)


def validate_dataset_stage(config: PipelineConfig) -> pd.DataFrame:
    ensure_run_dirs(config)
    dump_config(config, config.run_dir / "config_used.yaml")
    frame = load_dataset(config)
    report = validate_dataset(frame, config)
    write_validation_report(report, config.run_dir / "dataset_validation.json")
    frame.to_csv(config.run_dir / "validated_dataset.csv", index=False)
    return frame


def split_dataset_stage(config: PipelineConfig) -> pd.DataFrame:
    path = config.run_dir / "validated_dataset.csv"
    frame = pd.read_csv(path) if path.exists() else validate_dataset_stage(config)
    split_frame = deterministic_split(frame, config)
    split_frame.to_csv(config.run_dir / "splits" / "dataset_with_splits.csv", index=False)
    return split_frame


def train_predictors_stage(config: PipelineConfig, model: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame_path = config.run_dir / "splits" / "dataset_with_splits.csv"
    frame = pd.read_csv(frame_path) if frame_path.exists() else split_dataset_stage(config)
    if model:
        config = type(config)(
            run_name=config.run_name,
            input_path=config.input_path,
            output_dir=config.output_dir,
            architecture_id_column=config.architecture_id_column,
            architecture_column=config.architecture_column,
            feature_columns=config.feature_columns,
            target_columns=config.target_columns,
            objectives=config.objectives,
            split=config.split,
            seed=config.seed,
            predictors=(model,),
            zero_cost_proxies=config.zero_cost_proxies,
            nsga3_population_size=config.nsga3_population_size,
            nsga3_generations=config.nsga3_generations,
            nsga3_archive_size=config.nsga3_archive_size,
            csv_has_header=config.csv_has_header,
            csv_column_names=config.csv_column_names,
            plot=config.plot,
        )
    metrics, predictions = train_predictors(frame, config)
    metrics.to_csv(config.run_dir / "metrics" / "predictor_metrics.csv", index=False)
    predictions.to_csv(config.run_dir / "predictions" / "predictor_predictions.csv", index=False)
    return metrics, predictions


def evaluate_predictors_stage(config: PipelineConfig) -> pd.DataFrame:
    metrics_path = config.run_dir / "metrics" / "predictor_metrics.csv"
    if metrics_path.exists():
        return pd.read_csv(metrics_path)
    metrics, _ = train_predictors_stage(config)
    return metrics


def zero_cost_stage(config: PipelineConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame_path = config.run_dir / "splits" / "dataset_with_splits.csv"
    frame = pd.read_csv(frame_path) if frame_path.exists() else split_dataset_stage(config)
    scores, metrics = compute_zero_cost(frame, config)
    scores.to_csv(config.run_dir / "predictions" / "zero_cost_scores.csv", index=False)
    metrics.to_csv(config.run_dir / "metrics" / "zero_cost_metrics.csv", index=False)
    return scores, metrics


def nsga3_stage(config: PipelineConfig) -> pd.DataFrame:
    frame_path = config.run_dir / "splits" / "dataset_with_splits.csv"
    frame = pd.read_csv(frame_path) if frame_path.exists() else split_dataset_stage(config)
    selected = nsga3_style_selection(frame, config)
    selected.to_csv(config.run_dir / "pareto_fronts" / "nsga3_selection.csv", index=False)
    return selected


def generate_reports_stage(config: PipelineConfig) -> None:
    summary = {
        "run_name": config.run_name,
        "dataset_validation": str(config.run_dir / "dataset_validation.json"),
        "predictor_metrics": str(config.run_dir / "metrics" / "predictor_metrics.csv"),
        "zero_cost_metrics": str(config.run_dir / "metrics" / "zero_cost_metrics.csv"),
        "nsga3_selection": str(config.run_dir / "pareto_fronts" / "nsga3_selection.csv"),
    }
    (config.run_dir / "logs" / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if not config.plot:
        return

    predictions_path = config.run_dir / "predictions" / "predictor_predictions.csv"
    if predictions_path.exists():
        predictions = pd.read_csv(predictions_path)
        plot_predicted_vs_true(
            predictions[predictions["split"] == "test"],
            config.target_columns[0],
            config.run_dir / "figures" / "predicted_vs_true.png",
        )

    pareto_path = config.run_dir / "pareto_fronts" / "nsga3_selection.csv"
    if pareto_path.exists() and len(config.objectives) >= 2:
        selected = pd.read_csv(pareto_path)
        plot_pareto(
            selected,
            config.objectives[1].column,
            config.objectives[0].column,
            config.run_dir / "figures" / "nsga3_selection.png",
        )


def run_pipeline(config: PipelineConfig) -> PipelineConfig:
    validate_dataset_stage(config)
    split_dataset_stage(config)
    train_predictors_stage(config)
    zero_cost_stage(config)
    nsga3_stage(config)
    generate_reports_stage(config)
    return config


def run_pipeline_from_config(config_path: str | Path) -> PipelineConfig:
    config = load_config(config_path)
    return run_pipeline(config)
