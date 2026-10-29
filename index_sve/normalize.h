#pragma once
#include <torch/extension.h>
#include <vector>

struct IndexSpec {
  at::Tensor transposed_self;
  std::vector<int64_t> indexed_dims;
  std::vector<int64_t> indexed_sizes;
  std::vector<int64_t> indexed_strides;
  std::vector<at::Tensor> indices;
  std::vector<int64_t> broadcast_shape;
  int64_t B = 0, K = 0, A = 0;
  std::vector<int64_t> row_off;
  std::vector<int64_t> col_off;
  bool after_contiguous = true;
  // R10：非连续 out 的行块直写资格（index_impl 经 rowcopy_direct_ok 置位）。
  // 置位后 index_into 只允许走纯块拷贝内核（rowcopy_raw / rowcopy_sve），
  // 绝不能落入 col_off 类路径（块内行主序放置对置换 out 是错位写入）。
  bool rowcopy_direct = false;
  std::vector<int64_t> out_shape;
};

IndexSpec normalize_index(const at::Tensor& self,
                          const std::vector<c10::optional<at::Tensor>>& indices);
