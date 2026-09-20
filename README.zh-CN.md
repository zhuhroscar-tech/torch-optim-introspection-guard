[![English](https://img.shields.io/badge/English-555555?style=flat)](README.md) [![简体中文](https://img.shields.io/badge/简体中文-555555?style=flat)](README.zh-CN.md)

# torch-optim-introspection-guard

检查 PyTorch 在读取 optimizer state 时是否改变了下一步训练结果，并提供一个用于恢复 optimizer state 的 wrapper。项目针对 [pytorch/pytorch#164929](https://github.com/pytorch/pytorch/issues/164929) 描述的行为，并不排查所有训练结果不一致的问题。

`get_optimizer_state_dict()` 可能通过一次学习率为零的更新来初始化 optimizer。对于依赖 step counter 的算法，即使参数没有立即变化，后续更新仍可能受到影响。CLI 会在本机安装的 PyTorch 上比较 baseline、未保护调用和保护调用，不按版本号直接判断是否存在问题。

## 安装与诊断

需要 Python 3.9+，以及提供 `torch.distributed.checkpoint.state_dict` 的 PyTorch。源码安装：

```bash
git clone https://github.com/zhuhroscar-tech/torch-optim-introspection-guard.git
cd torch-optim-introspection-guard
python3 -m venv .venv
source .venv/bin/activate
python -m pip install '.[torch]'
torch-optim-introspection-guard
torch-optim-introspection-guard --json
```

如果当前环境已经安装兼容的 PyTorch，可改为安装 `.`，不使用 extra。`--seed` 可指定随机种子，`--no-color` 可关闭终端颜色。

退出码：**0** 表示所有受测 optimizer 的保护调用结果均与 baseline 一致；**1** 表示至少一项保护检查失败；**2** 表示无法导入 PyTorch 或所需 checkpoint API。保护检查通过不等于成功复现上游问题，请另行查看 `any_bug_present`。

## Python API

使用已有的 model 和 optimizer：

```python
from torch.distributed.checkpoint.state_dict import StateDictOptions
from torch_optim_introspection_guard import safe_get_optimizer_state_dict

state = safe_get_optimizer_state_dict(
    model, optimizer, options=StateDictOptions(full_state_dict=True)
)
```

对于已有 state 的 optimizer，wrapper 会先深拷贝，再在调用后恢复（无论成功与否）；对于新建 optimizer，则在 `finally` 中清空调用过程中初始化的 state。

## 适用范围与限制

诊断使用 CPU tensor，覆盖 AdamW、Adam、NAdam、RAdam、RMSprop、带或不带 momentum 的 SGD、Adagrad 和 Adadelta。这不代表已验证 fused CUDA optimizer、自定义 optimizer 的副作用或大型分布式训练。

State 快照可能占用大量内存，大模型开销尚未做 benchmark。恢复操作在 `finally` 中执行（v0.1.1 修复），因此即使底层调用在修改 state 之后抛出异常，仍会先恢复/清空再让异常传播。请先在自己的训练环境中验证。

## 开发

```bash
python -m pip install -e '.[dev,torch]'
python -m pytest -v
```

[MIT 许可证](LICENSE)。
