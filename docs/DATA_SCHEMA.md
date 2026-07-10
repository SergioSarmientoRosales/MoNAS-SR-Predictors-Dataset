# Data Schema

The cleaned pipeline uses a headered CSV schema.

| Column | Type | Required | Role | Description |
| --- | --- | --- | --- | --- |
| `architecture_id` | string | yes | identifier | Stable row/architecture identifier. |
| `chromosome` | string/list | yes | feature | Encoded architecture list. |
| `params` | numeric | recommended | objective/feature | Number of model parameters; minimized. |
| `flops` | numeric | recommended | objective/feature | Floating-point operations; minimized. |
| `valid_psnr` | numeric | yes | target/objective | Validation PSNR or quality metric; maximized. |
| `train_psnr` | numeric | optional | target/metadata | Training PSNR, if available. |
| `model` | string | optional | metadata | Predictor or source model. |
| `seed` | integer | optional | metadata | Random seed or replicate. |

The historical `Datasets/*.csv` files appear headerless. To reuse them, either add headers or configure `csv_has_header: false` and `csv_column_names`.

Validation checks:

- Required columns exist.
- Numeric columns parse as numeric.
- Architecture encodings parse as numeric lists.
- Architecture encodings have consistent length.
- Duplicate architecture IDs are counted in `dataset_validation.json`.
- Objective directions are declared in config.
