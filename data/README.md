# Data Directory

Use this directory for local full datasets that should not be committed by default.

The cleaned pipeline expects a CSV with columns documented in `docs/DATA_SCHEMA.md`. The toy dataset is available at `examples/example_architecture_dataset.csv`.

Historical committed datasets remain in `Datasets/`:

- `1000_reduced_models_psnr_result.csv`
- `1193_P_Trained_models_psnr_result.csv`
- `541_P_Trained_model_psnr_results.csv`

These legacy CSVs appear to be headerless and use architecture encoding plus PSNR columns. The cleaned configs use a documented headered schema for reproducible scripts.
