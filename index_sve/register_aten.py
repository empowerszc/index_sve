"""兼容层：aten::index.Tensor 的 CPU 覆盖已改由 C++ 扩展加载时完成
（op.cpp 的 TORCH_LIBRARY_IMPL(aten, CPU, ...)），避免 Python 覆盖层每调用一次的
解释器往返。install() 保留为空实现，供旧代码/文档调用。
"""
import torch


def install() -> None:
    # 覆盖已由 C++ 扩展在加载时通过 TORCH_LIBRARY_IMPL(aten, CPU) 完成；Python 侧
    # 注册会多一层解释器往返（小 K 每调用 ~10-20us）。保留函数以兼容旧调用。
    return None
