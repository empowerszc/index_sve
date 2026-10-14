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
  std::vector<int64_t> out_shape;
};

IndexSpec normalize_index(const at::Tensor& self,
                          const std::vector<c10::optional<at::Tensor>>& indices);
