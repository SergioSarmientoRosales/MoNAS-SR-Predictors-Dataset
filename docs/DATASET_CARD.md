# Dataset Card

## Dataset Name

MoNAS-SR Predictors Dataset.

## Purpose

Encoded architecture dataset and experimental artifacts for predictor-assisted multi-objective neural architecture search in single image super-resolution / super-resolution image restoration.

## Domain

Single Image Super-Resolution (SISR) / SRIR.

## Contents

The repository contains:

- Headerless legacy architecture/PSNR CSV files in `Datasets/`.
- Seed-level output populations for SVR, Extra Trees, XGBoost, and SynFlow in `Seeds/`.
- Final median-seed populations and filtered fronts in `Final Population in Median Seeds/`.
- Trained model artifacts in `Codes/` and `Final Population in Median Seeds/Neural Nets trained/`.
- A small documented toy dataset in `examples/example_architecture_dataset.csv`.

## Architecture Encoding

Architectures are represented as list-like integer chromosomes, for example:

```text
[1, 2, 3, 4, 1, 0, 2, 5]
```

The exact operation mapping for every position is not fully documented in the original repository and is marked as TBD for full scientific reuse.

## Targets and Objectives

Known or inferred columns include:

- PSNR-like quality values, usually maximized.
- Parameters, minimized.
- FLOPs, minimized.
- Predictor scores or zero-cost proxy scores, direction depends on method.

## Generation Process

The repository indicates that architectures were evaluated or estimated using predictor-assisted NAS, zero-cost SynFlow runs, and NSGA-III search scripts. Full generation metadata and training budget details remain to be documented for complete reuse.

## Splits

No canonical split file was found in the original repository. The cleaned pipeline creates deterministic train/validation/test splits using a configurable seed.

## Known Limitations

- Some legacy scripts contain local or Colab paths.
- Headerless legacy CSVs require schema assumptions.
- The clean `synflow_like` example is an interface smoke test, not the true TensorFlow SynFlow computation.
- The canonical citation is the 2026 Swarm and Evolutionary Computation article listed below and in `CITATION.cff`.

## Citation

Sarmiento-Rosales, S., Hernández, V. A. S., & Monroy, R. (2026). Evolutionary Neural Architecture Search for Super-Resolution: Benchmarking SynFlow and model-based predictors. Swarm and Evolutionary Computation, 100, 102236.
