# Reproduction Guide

## Fresh Clone

```bash
git clone https://github.com/SergioSarmientoRosales/MoNAS-SR-Predictors-Dataset.git
cd MoNAS-SR-Predictors-Dataset
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
pytest tests/
python scripts/run_pipeline.py --config configs/toy_example.yaml
```

On macOS/Linux, activate with:

```bash
source .venv/bin/activate
```

## Stage Commands

```bash
python scripts/01_validate_dataset.py --config configs/toy_example.yaml
python scripts/02_train_predictors.py --config configs/toy_example.yaml
python scripts/03_evaluate_predictors.py --config configs/toy_example.yaml
python scripts/04_compute_zero_cost.py --config configs/toy_example.yaml
python scripts/05_run_nsga3.py --config configs/toy_example.yaml
python scripts/06_generate_reports.py --config configs/toy_example.yaml
```

## Full Dataset

Use the templates in `configs/`, add documented headers or `csv_column_names`, and place local full data under `data/`.
