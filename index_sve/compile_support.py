"""torch.compile 的 opt-in 支持（HPC_INDEX_COMPILE=1）。

默认关闭。开启后（须在 import index_sve 前设置环境变量）：

1. 为 `hpc_index::index` / `index_out` 注册 fake，修复 FakeTensor/torch.export；
2. 把 Inductor 的 `aten.index.Tensor` lowering 永久替换为 fallback，使编译产物
   在运行时回到 dispatcher，从而命中本算子；
3. 把 Inductor 磁盘缓存切到独立子目录——缓存 key 不含 lowering 注册信息，若复用
   未打补丁时生成的缓存，补丁会被静默绕过。

风险：第 2 步依赖 `torch._inductor` 私有 API，已在 torch 2.10 验证；版本不符时
直接报错而不是静默失效。默认不开启，避免改变用户的编译缓存行为。
"""
import os

import torch

_ENABLED = False


def enabled() -> bool:
    return _ENABLED


def _fake_index(self, indices):
    # 整数下标形状可静态推导。bool/byte mask 的输出长度依赖数据（nonzero），
    # 本 fake 不覆盖；torch.compile 走 aten::index fallback，由 Inductor/aten
    # 自己的 mask 逻辑处理（见 torch-compile.md §6）。
    if any(
        i is not None and i.dtype in (torch.bool, torch.uint8) for i in indices
    ):
        raise NotImplementedError(
            "hpc_index::index fake impl does not support boolean masks "
            "(data-dependent output); use the aten::index fallback path"
        )
    shapes = [tuple(i.shape) for i in indices if i is not None]
    bshape = torch.broadcast_shapes(*shapes) if shapes else ()
    nidx = sum(i is not None for i in indices)
    return self.new_empty(bshape + tuple(self.shape[nidx:]))


def _register_fakes() -> None:
    @torch.library.register_fake("hpc_index::index")
    def _fake(self, indices):
        return _fake_index(self, indices)

    @torch.library.register_fake("hpc_index::index_out")
    def _fake_out(self, indices, out):
        out.resize_(_fake_index(self, indices).shape)
        return out


def _force_fallback() -> None:
    try:
        from torch._inductor import lowering
    except ImportError as e:
        raise RuntimeError(
            "HPC_INDEX_COMPILE=1 requires torch._inductor (torch 2.10 tested)"
        ) from e
    missing = [
        n for n in ("fallback_handler", "lowerings", "register_lowering")
        if not hasattr(lowering, n)
    ]
    if missing:
        raise RuntimeError(
            "HPC_INDEX_COMPILE=1: unsupported torch._inductor.lowering API "
            f"(missing {missing}); torch 2.10 tested"
        )
    op = torch.ops.aten.index.Tensor
    lowering.register_lowering(op)(lowering.fallback_handler(op))


def _redirect_cache_dir() -> str:
    # runtime_utils.cache_dir() 以 TORCHINDUCTOR_CACHE_DIR 为准，且惰性读取。
    from torch._inductor.runtime import runtime_utils

    base = runtime_utils.cache_dir()
    if base.rstrip(os.sep).endswith("hpc_index"):
        return base
    target = os.path.join(base, "hpc_index")
    os.makedirs(target, exist_ok=True)
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = target
    return target


def enable() -> None:
    global _ENABLED
    if _ENABLED:
        return
    if os.environ.get("HPC_INDEX_NO_OVERRIDE", "0") == "1":
        raise RuntimeError(
            "HPC_INDEX_COMPILE=1 conflicts with HPC_INDEX_NO_OVERRIDE=1: "
            "compile support only makes sense when the aten override is active"
        )
    _register_fakes()
    _force_fallback()
    _redirect_cache_dir()
    _ENABLED = True
