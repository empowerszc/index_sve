"""把 torch.ops.hpc_index.index 注册为 aten::index.Tensor 的 CPU 实现。

import index_sve 时自动调用 install()：此后 x[idx]、x[:, i] 等所有 CPU 张量的
高级索引调用都走本自定义算子（SVE512 gather + 线性化）。无需改任何调用点。
"""
import torch


def install() -> None:
    torch.library.impl("aten::index.Tensor", "CPU", torch.ops.hpc_index.index)
