# Examples

The examples use `examples/example_architecture_dataset.csv`, a tiny synthetic dataset with the same schema expected by the cleaned pipeline.

Run from the repository root:

```bash
python examples/minimal_dataset_example.py
python examples/train_predictor_example.py
python examples/zero_cost_example.py
python examples/nsga3_example.py
```

Outputs are written under `runs/examples/`.

The zero-cost example uses `synflow_like`, a deterministic toy proxy for CI and documentation. It is not a replacement for the historical TensorFlow SynFlow implementation in `Codes/synflow_compsr_nsga_iii.py`.
