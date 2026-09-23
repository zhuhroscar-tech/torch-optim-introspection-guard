[![English](https://img.shields.io/badge/English-555555?style=flat)](README.md) [![简体中文](https://img.shields.io/badge/简体中文-555555?style=flat)](README.zh-CN.md)

# torch-optim-introspection-guard

This standalone repository has been consolidated into [`torch-correctness-guards`](https://github.com/zhuhroscar-tech/torch-correctness-guards).

Use the umbrella package going forward:

```bash
python -m pip install 'torch-correctness-guards[torch]'
torch-guard run optim-introspection
```

Python API:

```python
from torch_correctness_guards import safe_get_optimizer_state_dict
```

The original functionality is preserved as the `optim-introspection` guard in the umbrella package, alongside the other PyTorch correctness diagnostics.

This repo is archived for history only. New fixes and releases happen in `torch-correctness-guards`.

[MIT license](LICENSE).
