# MoNAS-SR-Predictors-Dataset

Dataset and reproducible experimental scaffold for predictor-assisted multi-objective neural architecture search (MoNAS) in single image super-resolution.

This repository contains encoded super-resolution architecture data, predictor benchmark artifacts, SynFlow/zero-cost proxy outputs, and NSGA-III search scripts. The cleanup adds a lightweight, testable pipeline for public release and peer-review reproducibility while preserving the historical artifacts.

## What This Repository Provides

- Encoded architecture datasets for super-resolution NAS.
- Historical seed outputs for SVR, Extra Trees, XGBoost, and SynFlow.
- Historical trained models and final populations.
- A clean Python package for dataset loading, schema validation, predictor benchmarking, zero-cost proxy interface, and multi-objective selection.
- CLI scripts and toy examples that run from a fresh clone.

## Repository Structure

| Path | Purpose |
| --- | --- |
| `src/monas_sr_predictors/` | Clean package for dataset utilities, predictors, zero-cost proxies, metrics, Pareto utilities, and NSGA-III-style selection. |
| `configs/` | Toy and full-template YAML configs. |
| `scripts/` | Reproducible command-line stages. |
| `examples/` | Runnable toy dataset and example scripts. |
| `docs/` | Dataset card, schema, audit, predictor, zero-cost, NSGA-III, and reproduction docs. |
| `tests/` | Lightweight pytest suite. |
| `data/` | Local full dataset staging area. |
| `runs/` | Generated outputs from the clean pipeline. |
| `Codes/` | Historical notebooks/scripts/models. |
| `Datasets/` | Historical headerless CSV datasets. |
| `Seeds/` | Historical seed output populations. |
| `Final Population in Median Seeds/` | Historical final populations and trained networks. |

## Installation

```bash
git clone https://github.com/SergioSarmientoRosales/MoNAS-SR-Predictors-Dataset.git
cd MoNAS-SR-Predictors-Dataset
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

On macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Optional heavy dependencies for XGBoost and TensorFlow legacy scripts:

```bash
pip install -r requirements-optional.txt
```

## Quickstart

```bash
python scripts/run_pipeline.py --config configs/toy_example.yaml
```

Outputs are written to:

```text
runs/toy_example/
  config_used.yaml
  dataset_validation.json
  validated_dataset.csv
  splits/
  models/
  predictions/
  metrics/
  pareto_fronts/
  figures/
  logs/
```

Run tests:

```bash
pytest tests/
```

## Pipeline Stages

```bash
python scripts/01_validate_dataset.py --config configs/toy_example.yaml
python scripts/02_train_predictors.py --config configs/toy_example.yaml
python scripts/03_evaluate_predictors.py --config configs/toy_example.yaml
python scripts/04_compute_zero_cost.py --config configs/toy_example.yaml
python scripts/05_run_nsga3.py --config configs/toy_example.yaml
python scripts/06_generate_reports.py --config configs/toy_example.yaml
```

## Dataset Format

The clean pipeline expects a headered CSV. See `examples/example_architecture_dataset.csv`.

Default required columns:

| Column | Role | Direction |
| --- | --- | --- |
| `architecture_id` | identifier | n/a |
| `chromosome` | architecture encoding | n/a |
| `params` | feature/objective | minimize |
| `flops` | feature/objective | minimize |
| `valid_psnr` | target/objective | maximize |

Historical `Datasets/*.csv` files appear headerless. Use `docs/DATA_SCHEMA.md` and config fields `csv_has_header` / `csv_column_names` when adapting them.

## Predictor Benchmarks

Supported in the clean interface:

- `extra_trees`
- `svr`
- `random_forest`
- `xgboost` if optional dependency is installed

Train one model:

```bash
python scripts/02_train_predictors.py --config configs/toy_example.yaml --model extra_trees
```

Metrics include MAE, RMSE, R2, Pearson, and Spearman correlation. See `docs/PREDICTOR_BENCHMARKS.md`.

## Zero-Cost Proxies

Historical SynFlow outputs are preserved under `Seeds/SynFlow seeds/`, and the TensorFlow SynFlow script remains in `Codes/synflow_compsr_nsga_iii.py`.

The clean pipeline includes `synflow_like`, a deterministic toy proxy used only for examples and CI:

```bash
python scripts/04_compute_zero_cost.py --config configs/toy_example.yaml
```

It is not a replacement for true SynFlow. See `docs/ZERO_COST_PROXIES.md`.

## NSGA-III / Multi-Objective Selection

Historical NSGA-III scripts remain in `Codes/`. The clean pipeline adds a lightweight NSGA-III-style selection stage for reproducible smoke tests:

```bash
python scripts/05_run_nsga3.py --config configs/toy_example.yaml
```

See `docs/NSGA3_PIPELINE.md`.

## Examples

```bash
python examples/minimal_dataset_example.py
python examples/train_predictor_example.py
python examples/zero_cost_example.py
python examples/nsga3_example.py
```

## Reproduction

Use `configs/full_pipeline_template.yaml` for a full run. Place a documented full dataset under `data/`, update paths and schema fields, then run:

```bash
python scripts/run_pipeline.py --config configs/full_pipeline_template.yaml
```

## Troubleshooting

`ModuleNotFoundError: monas_sr_predictors`

Run `pip install -e .`, or run scripts from the repository root.

`Missing required columns`

Check `docs/DATA_SCHEMA.md` and your YAML config. Headerless legacy CSVs require explicit `csv_column_names`.

`xgboost is optional`

Install `requirements-optional.txt` only when reproducing the XGBoost benchmark.

Legacy scripts fail on paths

Some original files contain Colab or local paths. Prefer the clean scripts for reproducible examples, or adapt paths in `Codes/` deliberately.

## Citation

```bibtex
@misc{SarmientoRosales2026MoNASSRPredictorsDataset,
  title = {MoNAS-SR-Predictors-Dataset},
  author = {Sarmiento-Rosales, Sergio},
  year = {2026},
  note = {Dataset and framework for predictor-assisted multi-objective NAS in super-resolution. Publication venue and DOI TBD.}
}
```

## License

This repository includes a license file. See `LICENSE`.

## Contact

Open an issue on GitHub for questions, reproducibility problems, or dataset documentation gaps:

https://github.com/SergioSarmientoRosales/MoNAS-SR-Predictors-Dataset/issues
