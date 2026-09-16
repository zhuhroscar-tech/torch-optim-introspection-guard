[![English](https://img.shields.io/badge/English-555555?style=flat)](README.md) [![简体中文](https://img.shields.io/badge/简体中文-555555?style=flat)](README.zh-CN.md)

# torch-optim-introspection-guard

Check whether PyTorch optimizer-state introspection changes the next training step, and use a wrapper intended to preserve optimizer state around that call. This targets the behavior described in [pytorch/pytorch#164929](https://github.com/pytorch/pytorch/issues/164929), not every possible source of training divergence.

`get_optimizer_state_dict()` can initialize a fresh optimizer with a zero-learning-rate step. For step-dependent optimizers, advancing the counter can still change subsequent updates. The CLI compares baseline, unguarded, and guarded runs on your installed PyTorch build rather than assuming a particular version is affected.

## Install and diagnose

Requires Python 3.9+ and a PyTorch build exposing `torch.distributed.checkpoint.state_dict`. From source:

```bash
git clone https://github.com/zhuhroscar-tech/torch-optim-introspection-guard.git
cd torch-optim-introspection-guard
python3 -m venv .venv
source .venv/bin/activate
python -m pip install '.[torch]'
torch-optim-introspection-guard
torch-optim-introspection-guard --json
```

If you already manage a compatible PyTorch installation, install `.` without the extra. Use `--seed` for a reproducible probe and `--no-color` for plain text.

Exit codes: **0** means the guard matched the baseline for all tested optimizers, **1** means at least one guard check failed, and **2** means PyTorch or its required checkpoint API could not be imported. A successful guard check does not mean the upstream bug was reproduced; inspect `any_bug_present` separately.

## Python API

With your existing model and optimizer:

```python
from torch.distributed.checkpoint.state_dict import StateDictOptions
from torch_optim_introspection_guard import safe_get_optimizer_state_dict

state = safe_get_optimizer_state_dict(
    model, optimizer, options=StateDictOptions(full_state_dict=True)
)
```

The wrapper deep-copies existing optimizer state and reloads it after a successful call; for a fresh optimizer, it clears the initialized state instead.

## Scope and limitations

The probe covers AdamW, Adam, NAdam, RAdam, RMSprop, SGD with and without momentum, Adagrad, and Adadelta using CPU tensors. It is not validation of fused CUDA optimizers, custom optimizer side effects, or large distributed jobs. Snapshotting can consume substantial memory; large-model overhead has not been benchmarked. Restoration is not in a `finally` block, so an exception in the underlying call can leave mutations behind. Test your own training setup before relying on the wrapper.

## Development

```bash
python -m pip install -e '.[dev,torch]'
python -m pytest -v
```

[MIT license](LICENSE).
