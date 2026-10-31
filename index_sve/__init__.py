from . import _hpc_index  # noqa: F401  (dlopen 时注册 torch.ops.hpc_index)
from .register_aten import install as _install_aten  # noqa: F401
import torch
import os

OVERRIDE_ACTIVE = os.environ.get("HPC_INDEX_NO_OVERRIDE", "0") != "1"

if OVERRIDE_ACTIVE:
    _install_aten()  # 直接覆盖 aten::index.Tensor（CPU）；HPC_INDEX_NO_OVERRIDE=1 跳过（基准对拍用）

if os.environ.get("HPC_INDEX_COMPILE", "0") == "1":
    from .compile_support import enable as _enable_compile
    _enable_compile()  # opt-in：torch.compile fallback + fake + 独立 inductor 缓存

def index(self, indices):
    indices = list(indices)
    return torch.ops.hpc_index.index(self, indices)
