# Zero-Cost Proxies

The historical repository includes SynFlow output populations under `Seeds/SynFlow seeds/` and a TensorFlow-based script in `Codes/synflow_compsr_nsga_iii.py`.

The cleaned lightweight pipeline includes `synflow_like` only as a deterministic example proxy. It exists so tests, CI, and documentation can exercise the zero-cost proxy interface without GPU, image data, or TensorFlow.

Important:

- `synflow_like` is not a scientific replacement for true SynFlow.
- True SynFlow reproduction should use the legacy script after replacing local dataset paths and documenting the image data location.
- Proxy direction and interpretation must be documented before using full results in a paper table.

Command:

```bash
python scripts/04_compute_zero_cost.py --config configs/toy_example.yaml
```
