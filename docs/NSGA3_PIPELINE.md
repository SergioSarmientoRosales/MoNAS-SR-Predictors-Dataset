# NSGA-III / Multi-Objective Selection

The original repository contains heavy NSGA-III scripts in `Codes/` that are coupled to TensorFlow, trained predictors, image data, and local paths.

The cleaned pipeline adds a lightweight `nsga3_style_selection` stage for smoke tests and reproducible examples. It:

- Converts configured objectives to minimization values.
- Extracts the non-dominated set.
- Uses deterministic reference directions to select a small archive.
- Saves `pareto_fronts/nsga3_selection.csv`.

This stage preserves the public-reproducibility intent but is not a full replacement for the original search implementation.

Command:

```bash
python scripts/05_run_nsga3.py --config configs/toy_example.yaml
```

For full scientific reproduction, review and adapt:

- `Codes/Model_based_compsr_nsga_iii.py`
- `Codes/synflow_compsr_nsga_iii.py`
