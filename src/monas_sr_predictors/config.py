from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ObjectiveSpec:
    column: str
    direction: str

    def __post_init__(self) -> None:
        direction = self.direction.lower()
        if direction not in {"minimize", "maximize"}:
            raise ValueError("Objective direction must be 'minimize' or 'maximize'.")
        object.__setattr__(self, "direction", direction)


@dataclass(frozen=True)
class SplitConfig:
    train: float = 0.6
    validation: float = 0.2
    test: float = 0.2

    def __post_init__(self) -> None:
        total = self.train + self.validation + self.test
        if abs(total - 1.0) > 1e-8:
            raise ValueError("Train/validation/test split ratios must sum to 1.")


@dataclass(frozen=True)
class PipelineConfig:
    run_name: str
    input_path: Path
    output_dir: Path
    architecture_id_column: str
    architecture_column: str
    feature_columns: tuple[str, ...]
    target_columns: tuple[str, ...]
    objectives: tuple[ObjectiveSpec, ...]
    split: SplitConfig = field(default_factory=SplitConfig)
    seed: int = 1
    predictors: tuple[str, ...] = ("extra_trees", "svr")
    zero_cost_proxies: tuple[str, ...] = ("synflow_like",)
    nsga3_population_size: int = 12
    nsga3_generations: int = 4
    nsga3_archive_size: int = 6
    csv_has_header: bool = True
    csv_column_names: tuple[str, ...] | None = None
    plot: bool = True

    @property
    def run_dir(self) -> Path:
        return self.output_dir / self.run_name


def _as_path(base_dir: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base_dir / path).resolve()


def _objectives(raw: Any) -> tuple[ObjectiveSpec, ...]:
    if not raw:
        raise ValueError("At least one objective must be configured.")
    return tuple(ObjectiveSpec(str(item["column"]), str(item["direction"])) for item in raw)


def load_config(path: str | Path) -> PipelineConfig:
    config_path = Path(path).resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    base_dir = config_path.parent.parent
    split_raw = raw.get("split", {})

    return PipelineConfig(
        run_name=str(raw.get("run_name", config_path.stem)),
        input_path=_as_path(base_dir, raw["input_path"]),
        output_dir=_as_path(base_dir, raw.get("output_dir", "runs")),
        architecture_id_column=str(raw.get("architecture_id_column", "architecture_id")),
        architecture_column=str(raw.get("architecture_column", "chromosome")),
        feature_columns=tuple(str(value) for value in raw.get("feature_columns", [])),
        target_columns=tuple(str(value) for value in raw.get("target_columns", ["valid_psnr"])),
        objectives=_objectives(raw.get("objectives")),
        split=SplitConfig(
            train=float(split_raw.get("train", 0.6)),
            validation=float(split_raw.get("validation", 0.2)),
            test=float(split_raw.get("test", 0.2)),
        ),
        seed=int(raw.get("seed", 1)),
        predictors=tuple(str(value) for value in raw.get("predictors", ["extra_trees", "svr"])),
        zero_cost_proxies=tuple(str(value) for value in raw.get("zero_cost_proxies", ["synflow_like"])),
        nsga3_population_size=int(raw.get("nsga3", {}).get("population_size", 12)),
        nsga3_generations=int(raw.get("nsga3", {}).get("generations", 4)),
        nsga3_archive_size=int(raw.get("nsga3", {}).get("archive_size", 6)),
        csv_has_header=bool(raw.get("csv_has_header", True)),
        csv_column_names=tuple(raw["csv_column_names"]) if raw.get("csv_column_names") else None,
        plot=bool(raw.get("plot", True)),
    )


def dump_config(config: PipelineConfig, path: str | Path) -> None:
    payload = {
        "run_name": config.run_name,
        "input_path": str(config.input_path),
        "output_dir": str(config.output_dir),
        "architecture_id_column": config.architecture_id_column,
        "architecture_column": config.architecture_column,
        "feature_columns": list(config.feature_columns),
        "target_columns": list(config.target_columns),
        "objectives": [
            {"column": objective.column, "direction": objective.direction}
            for objective in config.objectives
        ],
        "split": {
            "train": config.split.train,
            "validation": config.split.validation,
            "test": config.split.test,
        },
        "seed": config.seed,
        "predictors": list(config.predictors),
        "zero_cost_proxies": list(config.zero_cost_proxies),
        "nsga3": {
            "population_size": config.nsga3_population_size,
            "generations": config.nsga3_generations,
            "archive_size": config.nsga3_archive_size,
        },
        "csv_has_header": config.csv_has_header,
        "csv_column_names": list(config.csv_column_names) if config.csv_column_names else None,
        "plot": config.plot,
    }
    Path(path).write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
