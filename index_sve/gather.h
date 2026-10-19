#pragma once
#include "normalize.h"

void gather_index(const IndexSpec& spec, const at::Tensor& idx, at::Tensor& out,
                  bool non_temporal = false);  // non_temporal 供 Task 8 的 HPC_INDEX_NT 开关用
void gather_direct(const IndexSpec& spec, at::Tensor& out);  // K<1024 快速路径
void gather_rowcopy_raw(const IndexSpec& spec, at::Tensor& out, bool non_temporal = false);  // M1 单索引大块 skip-linearize
