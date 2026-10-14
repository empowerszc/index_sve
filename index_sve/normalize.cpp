#include "normalize.h"
#include <ATen/ExpandUtils.h>
#include <ATen/Functions.h>
#include <c10/util/Exception.h>
#include <c10/util/irange.h>

namespace {

bool is_defined(const c10::optional<at::Tensor>& t) {
  return t.has_value() && t->defined();
}

// 复刻 IndexingUtils.h checkIndexTensorTypes(allow_int=true)
void check_index_types(const std::vector<c10::optional<at::Tensor>>& indices) {
  for (const auto& t : indices) {
    if (t.has_value() && t->defined()) {
      auto st = t->scalar_type();
      if (st != at::kLong && st != at::kInt && st != at::kByte && st != at::kBool) {
        TORCH_CHECK_INDEX(false,
            "tensors used as indices must be long, int, byte or bool tensors");
      }
    }
  }
}

// 复刻 IndexingUtils.h expandTensors（ensure_same_device=false 简化：self CPU）
std::vector<at::Tensor> expand_tensors(
    const at::Tensor& self,
    const std::vector<c10::optional<at::Tensor>>& indices) {
  std::vector<at::Tensor> result;
  for (const auto& index_opt : indices) {
    if (!is_defined(index_opt)) {
      result.emplace_back();
      continue;
    }
    const at::Tensor& index = *index_opt;
    if (index.scalar_type() == at::kByte || index.scalar_type() == at::kBool) {
      if (index.scalar_type() == at::kByte) {
        TORCH_WARN("indexing with dtype torch.uint8 is now deprecated,"
                   " please use a dtype torch.bool instead.");
      }
      int64_t srcIdx = static_cast<int64_t>(result.size());
      for (const auto j : c10::irange(index.dim())) {
        if (index.size(j) != self.size(srcIdx + j)) {
          TORCH_CHECK_INDEX(false,
              "The shape of the mask ", index.sizes(),
              " at index ", j,
              " does not match the shape of the indexed tensor ", self.sizes(),
              " at index ", srcIdx + j);
        }
      }
      at::Tensor nonzero = index.nonzero();
      for (const auto j : c10::irange(index.dim())) {
        result.emplace_back(nonzero.select(1, j));
      }
    } else {
      result.emplace_back(index);
    }
  }
  return result;
}

bool has_contiguous_subspace(const std::vector<at::Tensor>& tl) {
  auto start = std::find_if(tl.begin(), tl.end(), [](const at::Tensor& t) { return t.defined(); });
  if (start == tl.end()) return true;
  auto stop = std::find_if(tl.rbegin(), tl.rend(), [](const at::Tensor& t) { return t.defined(); });
  return std::find_if(start, stop.base(), [](const at::Tensor& t) { return !t.defined(); }) == stop.base();
}

// 复刻 IndexingUtils.h transposeToFront
std::pair<at::Tensor, std::vector<at::Tensor>> transpose_to_front(
    const at::Tensor& self, const std::vector<at::Tensor>& indices) {
  std::vector<int64_t> dims;
  std::vector<at::Tensor> out_indices;
  dims.reserve(self.dim());
  for (const auto i : c10::irange(self.dim())) {
    if (indices[i].defined()) {
      dims.push_back(i);
      out_indices.emplace_back(indices[i]);
    }
  }
  for (const auto i : c10::irange(self.dim())) {
    if (!indices[i].defined()) {
      dims.push_back(i);
      out_indices.emplace_back();
    }
  }
  return {self.permute(dims), std::move(out_indices)};
}

std::vector<int64_t> decode_offsets(int64_t num, const std::vector<int64_t>& sizes,
                                    const std::vector<int64_t>& strides) {
  // 把 [0,num) 平铺编号解码为 sizes 各维坐标 * strides 的元素偏移
  std::vector<int64_t> off(num, 0);
  std::vector<int64_t> coord(sizes.size(), 0);
  std::vector<int64_t> stride_factor(sizes.size(), 1);
  for (int64_t i = sizes.size() - 2; i >= 0; --i) {
    stride_factor[i] = stride_factor[i + 1] * sizes[i + 1];
  }
  for (int64_t p = 0; p < num; ++p) {
    int64_t r = p;
    int64_t acc = 0;
    for (size_t d = 0; d < sizes.size(); ++d) {
      int64_t c = (sizes[d] > 0) ? (r / stride_factor[d]) % sizes[d] : 0;
      acc += c * strides[d];
      if (sizes[d] > 0) r %= stride_factor[d];
    }
    off[p] = acc;
  }
  return off;
}

}  // namespace

IndexSpec normalize_index(const at::Tensor& self,
                          const std::vector<c10::optional<at::Tensor>>& indices) {
  TORCH_CHECK(self.device().is_cpu(), "hpc_index: CPU only");
  check_index_types(indices);

  std::vector<at::Tensor> expanded = expand_tensors(self, indices);
  try {
    expanded = at::expand_outplace(expanded);  // 忽略 undefined，广播到公共形状
  } catch (std::exception&) {
    TORCH_CHECK_INDEX(false, "shape mismatch: indexing tensors could not be broadcast together with shapes");
  }
  while (expanded.size() < (size_t)self.dim()) {
    expanded.emplace_back();
  }
  at::Tensor tself = self;
  if (!has_contiguous_subspace(expanded)) {
    auto p = transpose_to_front(tself, expanded);
    tself = p.first;
    expanded = std::move(p.second);
  }

  IndexSpec spec;
  spec.transposed_self = tself;
  int64_t dims_before = 0, dims_after = 0, dims_indexed = 0;
  std::vector<int64_t> replacement_shape;
  for (const auto dim : c10::irange(expanded.size())) {
    if (!expanded[dim].defined()) {
      if (dims_indexed == 0) {
        dims_before++;
      } else {
        dims_after++;
      }
    } else {
      dims_indexed++;
      spec.indexed_dims.push_back(dim);
      spec.indexed_sizes.push_back(tself.size(dim));
      spec.indexed_strides.push_back(tself.stride(dim));
      replacement_shape = expanded[dim].sizes().vec();
      spec.indices.push_back(expanded[dim]);
    }
  }
  if (std::find(spec.indexed_sizes.begin(), spec.indexed_sizes.end(), 0) !=
          spec.indexed_sizes.end() &&
      std::find(replacement_shape.begin(), replacement_shape.end(), 0) ==
          replacement_shape.end()) {
    TORCH_CHECK_INDEX(false, "index is out of bounds for dimension with size 0");
  }

  spec.broadcast_shape = replacement_shape;
  spec.K = 1;
  for (int64_t s : replacement_shape) spec.K *= s;

  // 非转置时 before 维位于 [0, dims_before)，转置后 dims_before==0，两类布局统一处理
  std::vector<int64_t> before_sizes, after_sizes;
  for (const auto dim : c10::irange(expanded.size())) {
    if (dim < dims_before) {
      before_sizes.push_back(tself.size(dim));
    } else if (dim < dims_before + dims_indexed) {
      continue;
    } else {
      after_sizes.push_back(tself.size(dim));
    }
  }
  spec.B = 1;
  for (int64_t s : before_sizes) spec.B *= s;
  spec.A = 1;
  for (int64_t s : after_sizes) spec.A *= s;

  spec.out_shape = before_sizes;
  spec.out_shape.insert(spec.out_shape.end(), replacement_shape.begin(), replacement_shape.end());
  spec.out_shape.insert(spec.out_shape.end(), after_sizes.begin(), after_sizes.end());

  std::vector<int64_t> before_strides, after_strides;
  for (int64_t d = 0; d < dims_before; ++d) before_strides.push_back(tself.stride(d));
  for (int64_t d = dims_before + dims_indexed; d < (int64_t)expanded.size(); ++d) after_strides.push_back(tself.stride(d));
  spec.row_off = decode_offsets(spec.B, before_sizes, before_strides);
  spec.col_off = decode_offsets(spec.A, after_sizes, after_strides);

  spec.after_contiguous = true;
  for (int64_t c = 0; c < spec.A; ++c) {
    if (spec.col_off[c] != c) {
      spec.after_contiguous = false;
      break;
    }
  }
  return spec;
}
