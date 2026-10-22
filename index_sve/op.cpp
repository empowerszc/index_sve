#include <torch/extension.h>
#include "normalize.h"
#include "linearize.h"
#include "gather.h"
#include <ATen/MemoryOverlap.h>
#include <c10/util/Exception.h>
#include <atomic>
#include <cstdlib>

namespace {

std::vector<c10::optional<at::Tensor>> to_vec(
    const c10::List<c10::optional<at::Tensor>>& indices) {
  std::vector<c10::optional<at::Tensor>> vec;
  vec.reserve(indices.size());
  for (const auto& t : indices) vec.push_back(t);
  return vec;
}

bool non_temporal_enabled() {
  static const bool v = [] {
    const char* e = getenv("HPC_INDEX_NT");
    return e != nullptr && std::string(e) != "0";
  }();
  return v;
}

// 核心路由：out 已由调用方确定为正确 shape/dtype/device（functional 分配或 out 变体校验）。
void index_into(const at::Tensor& self, const IndexSpec& spec, at::Tensor& out) {
  const bool g_nt = non_temporal_enabled();

  if (spec.K < 1024) {                 // 小张量快速路径（spec §6 阈值）
    gather_direct(spec, out);
    return;
  }

  // M1 单索引大块 skip-linearize：单索引 + after_contiguous + A 大 + 值域小（S1 最热形态）。
  // 跳过 linearize，直接在 gather 阶段读原始下标算偏移（T2 阶段 1，无桶化）。
  if (spec.indices.size() == 1 && spec.after_contiguous && spec.A >= 4096 &&
      (out.scalar_type() == at::kFloat || out.scalar_type() == at::kDouble)) {
    gather_rowcopy_raw(spec, out, g_nt);
    return;
  }

  // T4 raw 小值域 1D 纯 gather：单索引 + B==1 + A==1（M3 keys[order]、M4 等）直接查表，
  // 完全跳过 linearize 的 broadcast 解码。
  if (spec.indices.size() == 1 && spec.B == 1 && spec.A == 1) {
    if (gather_bucket_pure_raw(spec, out)) return;
    // 大值域（>2048）时 1 线程走单遍标量（跳过 linearize）：SVE gather 单线程 MLP 低
    // + linearize 两遍开销，实测 1 线程 0.6-0.8x；单遍标量追平 aten。多线程仍走 SVE。
    if (at::get_num_threads() == 1) {
      gather_single_1d(spec, out);
      return;
    }
  }

  const bool sve_rowcopy_dtype =
      out.scalar_type() == at::kFloat || out.scalar_type() == at::kDouble;

  // f32 且 self.numel() >= 2^31 时 use_i32=false（idx 用 int64），gather_index 里
  // 4B SVE 分支要求 idx32 故不触发 → 自动落标量 dispatch_scalar（2^31 元素越界 int32）。
  const bool use_i32 = self.numel() < (1LL << 31) &&
      (out.scalar_type() == at::kFloat || out.scalar_type() == at::kInt);
  at::Tensor idx = at::empty({spec.K}, self.options().dtype(use_i32 ? at::kInt : at::kLong));

  // 1 线程单索引 A>1（rowcopy/scalar 短行）：跳过 linearize 走 fused 单遍。SVE
  // rowcopy 单线程 MLP 低，且 linearize 对单索引 1 线程开销占比大（rowcopy A=4 短行
  // 实测 linearize 占 80%）。A==1 单索引已在上方 gather_single_1d 短路；多索引/mask
  // 保持 linearize+SVE（fused 解码开销更大，实测更差）。
  if (at::get_num_threads() == 1 && spec.indices.size() == 1 && spec.A > 1) {
    gather_fused(spec, out);
    return;
  }
  if (spec.A == 1 ||
      (spec.after_contiguous && spec.A >= 4 && sve_rowcopy_dtype)) {
    linearize(spec, idx);
    gather_index(spec, idx, out, g_nt);
    return;
  }
  gather_fused(spec, out);
  return;
}

// 单 mask 融合快路径：一次扫描 bool/byte 掩码直接压实 + 行拷贝，跳过 nonzero 的中间
// int64 下标与第二次 gather。仅在 mask 覆盖 self 前 md 维、两者均连续、同 CPU 时启用。
std::atomic<uint64_t> g_mask_fused{0};

bool mask_fast_path(const at::Tensor& self,
                    const std::vector<c10::optional<at::Tensor>>& vec,
                    at::Tensor& out) {
  if (vec.size() != 1 || !vec[0].has_value() || !vec[0]->defined()) return false;
  const at::Tensor& mask = *vec[0];
  const auto st = mask.scalar_type();
  if (st != at::kBool && st != at::kByte) return false;
  if (!self.is_cpu() || !mask.is_cpu()) return false;
  const int64_t md = mask.dim();
  if (md < 1 || md > self.dim()) return false;
  if (!mask.is_contiguous() || !self.is_contiguous()) return false;
  for (const auto j : c10::irange(md)) {
    if (mask.size(j) != self.size(j)) {
      TORCH_CHECK_INDEX(false,
          "The shape of the mask ", mask.sizes(),
          " at index ", j,
          " does not match the shape of the indexed tensor ", self.sizes(),
          " at index ", j);
    }
  }
  if (st == at::kByte) {
    TORCH_WARN("indexing with dtype torch.uint8 is now deprecated,"
               " please use a dtype torch.bool instead.");
  }
  out = gather_mask(self, mask);
  g_mask_fused.fetch_add(1, std::memory_order_relaxed);
  return true;
}

}  // namespace

at::Tensor index_impl(const at::Tensor& self,
                      c10::List<c10::optional<at::Tensor>> indices) {
  auto vec = to_vec(indices);
  TORCH_CHECK_INDEX(!vec.empty(), "at least one index must be provided");
  TORCH_CHECK_INDEX(vec.size() <= (size_t)self.dim(),
      "too many indices for tensor of dimension ", self.dim(), " (got ", vec.size(), ")");
  at::Tensor out;
  if (mask_fast_path(self, vec, out)) return out;
  auto spec = normalize_index(self, vec);
  out = at::empty(spec.out_shape, self.options());
  index_into(self, spec, out);
  return out;
}

// aten::index.Tensor_out 的 CPU 实现：写入调用方提供的 out（形状不符时 resize），
// 非连续 out 计算到临时连续张量后 copy_。校验与 aten 的结构化 out 语义对齐。
at::Tensor index_out_impl(const at::Tensor& self,
                          c10::List<c10::optional<at::Tensor>> indices,
                          at::Tensor& out) {
  auto vec = to_vec(indices);
  TORCH_CHECK_INDEX(!vec.empty(), "at least one index must be provided");
  TORCH_CHECK_INDEX(vec.size() <= (size_t)self.dim(),
      "too many indices for tensor of dimension ", self.dim(), " (got ", vec.size(), ")");
  auto spec = normalize_index(self, vec);
  TORCH_CHECK(self.scalar_type() == out.scalar_type(),
      "index_out: self (", self.scalar_type(), ") and result (",
      out.scalar_type(), ") must have the same scalar type");
  TORCH_CHECK(self.device() == out.device(),
      "index_out: self and out must be on the same device");
  if (out.sizes() != spec.out_shape) out.resize_(spec.out_shape);
  at::assert_no_internal_overlap(out);
  at::assert_no_overlap(out, self);
  for (const auto& idx : spec.indices) at::assert_no_overlap(out, idx);
  if (out.is_contiguous()) {
    index_into(self, spec, out);
  } else {
    at::Tensor tmp = at::empty(spec.out_shape, self.options());
    index_into(self, spec, tmp);
    out.copy_(tmp);
  }
  return out;
}

TORCH_LIBRARY(hpc_index, m) {
  m.def("index(Tensor self, Tensor?[] indices) -> Tensor", &index_impl);
  m.def("index_out(Tensor self, Tensor?[] indices, Tensor(a!) out) -> Tensor(a!)",
        &index_out_impl);
}
TORCH_LIBRARY_IMPL(hpc_index, CPU, m) {
  m.impl("index", &index_impl);
  m.impl("index_out", &index_out_impl);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("index", [](const at::Tensor& self,
                    std::vector<c10::optional<at::Tensor>> indices) {
    static auto op = c10::Dispatcher::singleton()
        .findSchemaOrThrow("hpc_index::index", "")
        .typed<at::Tensor(const at::Tensor&, c10::List<c10::optional<at::Tensor>>)>();
    return op.call(self, c10::List<c10::optional<at::Tensor>>(indices));
  });
  m.def("linearize_spec", [](const at::Tensor& self,
                             std::vector<c10::optional<at::Tensor>> indices) {
    auto spec = normalize_index(self, indices);
    at::Tensor idx = at::empty({spec.K}, self.options().dtype(at::kLong));
    linearize(spec, idx);
    return idx;
  });
  m.def("debug_spec", [](const at::Tensor& self,
                         std::vector<c10::optional<at::Tensor>> indices) {
    auto spec = normalize_index(self, indices);
    return std::make_tuple(spec.transposed_self, spec.indexed_dims, spec.indexed_sizes,
                           spec.indexed_strides, spec.B, spec.K, spec.A, spec.out_shape,
                           spec.row_off, spec.col_off, spec.after_contiguous, spec.indices);
  });
  m.def("mask_fused_count", []() {
    return g_mask_fused.load(std::memory_order_relaxed);
  });
  m.def("idx_dtype_choice", [](int64_t numel, const std::string& dtype) {
    // 镜像 index_impl 的 use_i32 判定（仅供测试验证 f32 溢出回退逻辑）
    const bool is_4b = (dtype == "Float" || dtype == "Int");
    return (numel < (1LL << 31) && is_4b) ? 1 : 0;
  });
}
