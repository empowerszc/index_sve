#pragma once
#include "normalize.h"

void gather_index(const IndexSpec& spec, const at::Tensor& idx, at::Tensor& out,
                  bool non_temporal = false);  // non_temporal 供 Task 8 的 HPC_INDEX_NT 开关用
void gather_direct(const IndexSpec& spec, at::Tensor& out);  // K<1024 快速路径
void gather_rowcopy_raw(const IndexSpec& spec, at::Tensor& out, bool non_temporal = false);  // M1 单索引大块 skip-linearize
void gather_rowcopy_fused(const IndexSpec& spec, at::Tensor& out, bool non_temporal = false);  // 多索引 A>=4 skip-linearize
bool gather_strided_rows_raw(const IndexSpec& spec, at::Tensor& out, bool non_temporal = false);  // 非连续尾维 run 块拷贝
void gather_fused(const IndexSpec& spec, at::Tensor& out);  // T3 linearize+gather 融合单遍（scalar 小块）
bool gather_bucket_pure_raw(const IndexSpec& spec, at::Tensor& out);  // T4 raw：1D 纯 gather 跳 linearize 直接查表
void gather_single_1d(const IndexSpec& spec, at::Tensor& out);  // 1 线程单索引 1D 纯 gather 单遍（追平 aten）
void gather_single_1d_sve(const IndexSpec& spec, at::Tensor& out);  // T15-C：f32/i32+i64 下标多线程单遍 SVE
at::Tensor gather_mask(const at::Tensor& self, const at::Tensor& mask);  // 单 mask 单遍 compaction + 行拷贝
uint64_t strided_rows_count();
uint64_t rowcopy_raw_count();
