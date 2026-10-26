# error-geometry-routing

## Environment

uv project, Python 3.14, NVIDIA driver with CUDA ≥ 13.0 (torch `+cu130` from the PyTorch index, see
`pyproject.toml`; the wheels bundle their own CUDA runtime).

```
uv sync
uv run pytest tests
```

Run modules from the repository root with `uv run`, e.g. `uv run python -m sae.hooks`.

## SAE experiment

Plan: [PLAN.md](PLAN.md); code and run order: [sae/\_\_init\_\_.py](sae/__init__.py).
