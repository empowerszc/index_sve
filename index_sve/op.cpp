#include <torch/extension.h>
#include "normalize.h"
#include "linearize.h"
#include "gather.h"
#include "grain.h"
#include <ATen/MemoryOverlap.h>
#include <c10/util/Exception.h>
#include <atomic>
#include <cstring>
#include <cstdlib>
#ifdef _OPENMP
#include <omp.h>
#endif
#ifdef __ARM_FEATURE_SVE
#include <arm_sve.h>
#endif

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

void sync_omp_threads() {
#ifdef _OPENMP
  // 扩展用 BiSheng libomp，ATen 用 libgomp；torch.set_num_threads 只改后者。
  // 入口同步一次，使 torch.set_num_threads 对本内核的所有路径生效（T11）。
  omp_set_num_threads(static_cast<int>(at::get_num_threads()));
#endif
}

std::atomic<uint64_t> g_single_1d{0};
std::atomic<uint64_t> g_single_1d_sve{0};

// 核心路由：out 已由调用方确定为正确 shape/dtype/device（functional 分配或 out 变体校验）。
void index_into(const at::Tensor& self, const IndexSpec& spec, at::Tensor& out) {
  const bool g_nt = non_temporal_enabled();

  if (spec.K < 1024) {                 // 小张量快速路径（spec §6 阈值）
    gather_direct(spec, out);
    return;
  }

  // M1 单索引大块 skip-linearize：单索引 + after_contiguous + A 大 + 值域小（S1 最热形态）。
  // 跳过 linearize，直接在 gather 阶段读原始下标算偏移（T2 阶段 1，无桶化）。
  // T15：中等行宽 A<=64 且总行数 B*K<=65536 时放行（2D A=16 K=8192 曾落 linearize
  // 两遍，32/36t 0.73/0.76x）。大 K 不放行：linearize 的预校验偏移+无检查拷贝更快
  // （A/B 实测 raw 慢 12-32%：A=64 E=1M 8t +30%、A=16 K=200k 8t +32%、M2 (20000,64)
  // 8t +12%）——raw 逐行检查/负归一/v*stride 链在冷源（L3/DRAM）行拷贝下拖慢发射。
  if (spec.indices.size() == 1 && spec.after_contiguous &&
      spec.indices[0].is_contiguous() &&
      (spec.A >= 4096 || spec.A < 16 ||
       (spec.A <= 64 && spec.B * spec.K <= 65536)) &&
      spec.A > 1 &&
      (out.scalar_type() == at::kFloat || out.scalar_type() == at::kDouble)) {
    gather_rowcopy_raw(spec, out, g_nt);
    return;
  }

  // 非连续尾维：按 col_off 的连续 run 做块拷贝，仅当存在 >=4 的 run 时接管；
  // run=1 的纯转置继续走 fused，避免把逐元素伪装成“优化”。
  if (spec.indices.size() == 1 && !spec.after_contiguous && spec.A > 1 &&
      spec.indices[0].is_contiguous() &&
      (out.scalar_type() == at::kFloat || out.scalar_type() == at::kDouble)) {
    if (gather_strided_rows_raw(spec, out, g_nt)) return;
  }

  // 多索引 A>=4：跳过 linearize，在 rowcopy 循环里直接解码原始下标。
  if (spec.indices.size() > 1 && spec.after_contiguous && spec.A >= 4 &&
      at::get_num_threads() >= 16 &&
      (out.scalar_type() == at::kFloat || out.scalar_type() == at::kDouble)) {
    gather_rowcopy_fused(spec, out, g_nt);
    return;
  }

  // T4 raw 小值域 1D 纯 gather：单索引 + B==1 + A==1（M3 keys[order]、M4 等）直接查表，
  // 完全跳过 linearize 的 broadcast 解码。fast_1d 要求索引连续（raw 内核按 idx[k]
  // 平坦读；非连续索引走下面的 linearize 通用路径）。
  // 无 SVE 内核的 dtype（2B/1B/complex、f64+i32idx）在任何线程数都走单遍标量：
  // linearize 物化一遍偏移对这类标量 gather 纯属额外开销（实测 2-16t 0.5-0.9x）。
  // 有 SVE 内核时 1 线程仍用单遍（T5），多线程走 linearize+SVE。f32/i32 数据 +
  // i64 下标不走 linearize+SVE——那要 bucket 扫描+linearize+bucket 扫描+SVE 共
  // 4 遍 fork-join（T15 实测 K=65536 56→110µs）；T15 改为下方专用单遍 SVE 内核。
  if (spec.indices.size() == 1 && spec.B == 1 && spec.A == 1) {
    const at::Tensor& ind0 = spec.indices[0];
    const bool fast_1d = ind0.is_contiguous();
    if (fast_1d && gather_bucket_pure_raw(spec, out)) return;
    const bool idx32 = ind0.scalar_type() == at::kInt;
    const auto st = out.scalar_type();
#ifdef __ARM_FEATURE_SVE
    const bool hw_sve512 = (svcntb() == 64);
#else
    const bool hw_sve512 = false;
#endif
    const bool sve_ok = hw_sve512 &&
        ((idx32 && (st == at::kFloat || st == at::kInt)) ||
         (!idx32 && (st == at::kDouble || st == at::kLong)));
    // T15-C：f32/i32 数据 + 原始 i64 下标 + stride==1 的多线程 1D 纯 gather，源
    // >512KB（L2 实测 768KB，取其下保守界）时走专用单遍 SVE（读原始 i64、向量
    // 检查/归一/uzp1 收窄后复用 u32index gather），不经 linearize——gather_index
    // 前后的 bucket 值域扫描会多出两遍 fork-join。小源不走：实测每线程每元素
    // 标量 L1 命中 ~7.1ns vs 4B SVE gather ~18ns（微码/发射下限，双内核复现）；
    // 大源反转（L3/DRAM ~19 vs ~10ns，gather 的 8 路在飞负载摊薄延迟），见
    // degradation-analysis.md §5.9。
    if (fast_1d && !idx32 && hw_sve512 && at::get_num_threads() > 1 &&
        (st == at::kFloat || st == at::kInt) &&
        spec.indexed_strides[0] == 1 && spec.indexed_sizes[0] < (1LL << 32) &&
        self.numel() * self.element_size() > (512 << 10)) {
      g_single_1d_sve.fetch_add(1, std::memory_order_relaxed);
      gather_single_1d_sve(spec, out);
      return;
    }
    if (fast_1d && (!sve_ok || at::get_num_threads() == 1)) {
      g_single_1d.fetch_add(1, std::memory_order_relaxed);
      gather_single_1d(spec, out);
      return;
    }
  }

  // f32 且 self.numel() >= 2^31 时 use_i32=false（idx 用 int64），gather_index 里
  // 4B SVE 分支要求 idx32 故不触发 → 自动落标量 dispatch_scalar（2^31 元素越界 int32）。
  const bool use_i32 = self.numel() < (1LL << 31) &&
      (out.scalar_type() == at::kFloat || out.scalar_type() == at::kInt);
  at::Tensor idx = at::empty({spec.K}, self.options().dtype(use_i32 ? at::kInt : at::kLong));

  const bool sve_rowcopy_dtype =
      out.scalar_type() == at::kFloat || out.scalar_type() == at::kDouble;

  // 1 线程单索引 A>1：短行（A<4）用 fused 单遍，省掉 linearize；A>=4 连续块用
  // raw rowcopy 单遍（跳过 linearize，仍走 SVE 整块拷贝）。旧规则对所有 A>1 都用
  // fused，使 A=64 的 S1/P1 1 线程丢掉了 SVE rowcopy（1.06x vs >3.9x）。
  // A==1 单索引已在上方 gather_single_1d 短路；多索引/mask 保持 linearize+SVE。
  if (at::get_num_threads() == 1 && spec.indices.size() == 1 && spec.A > 1) {
    if (spec.A >= 4 && spec.after_contiguous &&
        spec.indices[0].is_contiguous() && sve_rowcopy_dtype) {
      gather_rowcopy_raw(spec, out, g_nt);
    } else {
      gather_fused(spec, out);
    }
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

std::atomic<uint64_t> g_small_single{0};
std::atomic<uint64_t> g_small_multi{0};

// 多整数下标（覆盖前 nidx 维）+ 小规模：直接按索引张量平坦顺序取行，绕过
// normalize 的 expand/transpose/vector 分配。要求所有索引同 shape 且连续，
// self 连续；广播、None、非连续一律回退通用路径。
bool small_multi_index(const at::Tensor& self,
                       const std::vector<c10::optional<at::Tensor>>& vec,
                       at::Tensor& out) {
  const int64_t nidx = vec.size();
  if (nidx < 2 || nidx > self.dim()) return false;
  if (!self.is_cpu() || !self.is_contiguous()) return false;

  const at::Tensor* first = nullptr;
  for (const auto& opt : vec) {
    if (!opt.has_value() || !opt->defined()) return false;
    const at::Tensor& ind = *opt;
    const auto st = ind.scalar_type();
    if (st != at::kLong && st != at::kInt) return false;
    if (!ind.is_cpu() || !ind.is_contiguous()) return false;
    if (first == nullptr) {
      first = &ind;
    } else if (ind.sizes() != first->sizes()) {
      return false;
    }
  }
  const int64_t n = first->numel();
  if (n > 65536) return false;

  int64_t A = 1;
  for (const auto d : c10::irange(nidx, self.dim())) A *= self.size(d);
  std::vector<int64_t> out_shape(first->sizes().begin(), first->sizes().end());
  out_shape.reserve(out_shape.size() + self.dim() - nidx);
  for (const auto d : c10::irange(nidx, self.dim())) out_shape.push_back(self.size(d));
  out = at::empty(out_shape, self.options());
  g_small_multi.fetch_add(1, std::memory_order_relaxed);
  if (n == 0 || A == 0) return true;

  std::vector<int64_t> ind_stride(nidx);
  int64_t stride = A;
  for (int64_t j = nidx - 1; j >= 0; --j) {
    ind_stride[j] = stride;
    stride *= self.size(j);
  }

  std::vector<const void*> ind_ptr(nidx);
  std::vector<int64_t> ind_size(nidx);
  std::vector<bool> ind_long(nidx);
  for (const auto j : c10::irange(nidx)) {
    ind_ptr[j] = vec[j]->data_ptr();
    ind_size[j] = self.size(j);
    ind_long[j] = (vec[j]->scalar_type() == at::kLong);
  }

  std::vector<int64_t> offs(n);
  int64_t first_off = 0;
  bool same = true;
  for (const auto p : c10::irange(n)) {
    int64_t off = 0;
    for (const auto j : c10::irange(nidx)) {
      int64_t v = ind_long[j]
          ? static_cast<const int64_t*>(ind_ptr[j])[p]
          : static_cast<const int32_t*>(ind_ptr[j])[p];
      const int64_t size = ind_size[j];
      TORCH_CHECK_INDEX(v >= -size && v < size,
          "index ", v, " is out of bounds for dimension ", j, " with size ", size);
      if (v < 0) v += size;
      off += v * ind_stride[j];
    }
    offs[p] = off;
    if (p == 0) {
      first_off = off;
    } else if (off != first_off) {
      same = false;
    }
  }

  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(at::kHalf, at::kBFloat16, at::kBool,
      out.scalar_type(), "hpc_index_small_multi", [&] {
    const scalar_t* src = static_cast<const scalar_t*>(self.data_ptr());
    scalar_t* dst = static_cast<scalar_t*>(out.data_ptr());
    const int64_t bytes = A * static_cast<int64_t>(sizeof(scalar_t));
    auto run = [&](int64_t lo, int64_t hi) {
      if (same) {
        if (A == 1) {
          std::fill(dst + lo, dst + hi, src[first_off]);
        } else {
          for (int64_t p = lo; p < hi; ++p) {
            std::memcpy(dst + p * A, src + first_off, bytes);
          }
        }
      } else if (A == 1) {
        for (int64_t p = lo; p < hi; ++p) dst[p] = src[offs[p]];
      } else {
        for (int64_t p = lo; p < hi; ++p) {
          std::memcpy(dst + p * A, src + offs[p], bytes);
        }
      }
    };
    if (n * A >= 2048) {
      at::parallel_for(0, n, adaptive_grain(n), run);
    } else {
      run(0, n);
    }
  });
  return true;
}

// small_single_index 的并行阈值：小 K 下 parallel_for 的任务开销超过收益，按总
// 拷贝元素数 n*A 而非 n 判断（2D A=16 K=4096 曾因只看 n 而串行，8-36t 掉到
// 0.55-0.66x）。
inline int64_t small_single_par_threshold() {
  // T15：高线程 fork-join 更贵，按线程数抬下限（smallk 38t：K=1024×A=16 并行
  // 37µs vs 串行 27µs，净亏 ~10µs；K=4096×A=16 的 65536 元素仍需并行）。
  return std::max<int64_t>(16384, 512 * at::get_num_threads());
}

// 单整数下标 + 末维 K<=4096：直接按 idx 平坦顺序取行，绕过 normalize 的向量分配/
// transpose 分析，也避开 gather_direct 内层对 at::Tensor 方法的重复访问。
bool small_single_index(const at::Tensor& self,
                        const std::vector<c10::optional<at::Tensor>>& vec,
                        at::Tensor& out) {
  if (vec.size() != 1 || !vec[0].has_value() || !vec[0]->defined()) return false;
  const at::Tensor& idx = *vec[0];
  const auto st = idx.scalar_type();
  if (st != at::kLong && st != at::kInt) return false;
  if (!self.is_cpu() || !idx.is_cpu()) return false;
  const int64_t K = idx.dim() == 0 ? 1 : idx.size(-1);
  if (K > 4096) return false;
  if (!self.is_contiguous() || !idx.is_contiguous()) return false;

  int64_t A = 1;
  for (const auto d : c10::irange(1, self.dim())) A *= self.size(d);
  std::vector<int64_t> out_shape(idx.sizes().begin(), idx.sizes().end());
  out_shape.reserve(out_shape.size() + self.dim() - 1);
  for (const auto d : c10::irange(1, self.dim())) out_shape.push_back(self.size(d));
  out = at::empty(out_shape, self.options());
  g_small_single.fetch_add(1, std::memory_order_relaxed);

  const int64_t n = idx.numel();
  if (n == 0) return true;
  const int64_t size0 = self.size(0);
  const bool idx_long = (st == at::kLong);
  const void* ip = idx.data_ptr();
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(at::kHalf, at::kBFloat16, at::kBool,
      out.scalar_type(), "hpc_index_small_single", [&] {
    const scalar_t* src = static_cast<const scalar_t*>(self.data_ptr());
    scalar_t* dst = static_cast<scalar_t*>(out.data_ptr());
    auto run = [&](int64_t lo, int64_t hi) {
      for (int64_t p = lo; p < hi; ++p) {
        int64_t v = idx_long ? static_cast<const int64_t*>(ip)[p]
                             : static_cast<const int32_t*>(ip)[p];
        TORCH_CHECK_INDEX(v >= -size0 && v < size0,
            "index ", v, " is out of bounds for dimension 0 with size ", size0);
        if (v < 0) v += size0;
        if (A == 0) continue;
        const scalar_t* sp = src + v * A;
        scalar_t* dp = dst + p * A;
        if (A >= 2) {
          // A=3 的 3 次迭代标量循环每行耗时 ~2x 于 memcpy（shortq 实测 35µs
          // vs 16µs；A=2 仅 ~1.1x），8/12B 小拷贝交给 memcpy 内联序列。
          std::memcpy(dp, sp, A * sizeof(scalar_t));
        } else {
          dp[0] = sp[0];
        }
      }
    };
    if (n * A >= small_single_par_threshold()) {
      at::parallel_for(0, n, adaptive_grain(n), run);
    } else {
      run(0, n);
    }
  });
  return true;
}

}  // namespace

at::Tensor index_impl(const at::Tensor& self,
                      const c10::List<c10::optional<at::Tensor>>& indices) {
  auto vec = to_vec(indices);
  TORCH_CHECK_INDEX(!vec.empty(), "at least one index must be provided");
  TORCH_CHECK_INDEX(vec.size() <= (size_t)self.dim(),
      "too many indices for tensor of dimension ", self.dim(), " (got ", vec.size(), ")");
  sync_omp_threads();
  at::Tensor out;
  if (mask_fast_path(self, vec, out)) return out;
  if (small_single_index(self, vec, out)) return out;
  if (small_multi_index(self, vec, out)) return out;
  auto spec = normalize_index(self, vec);
  out = at::empty(spec.out_shape, self.options());
  index_into(self, spec, out);
  return out;
}

// aten::index.Tensor_out 的 CPU 实现：写入调用方提供的 out（形状不符时 resize），
// 非连续 out 计算到临时连续张量后 copy_。校验与 aten 的结构化 out 语义对齐。
at::Tensor index_out_impl(const at::Tensor& self,
                          const c10::List<c10::optional<at::Tensor>>& indices,
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
  sync_omp_threads();
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
  // 只定义 schema，不挂 CompositeImplicitAutograd kernel：否则 PyTorch 拒绝为
  // 该 op 注册 fake/meta（torch.compile / torch.export 需要，见 compile_support.py）。
  m.def("index(Tensor self, Tensor?[] indices) -> Tensor");
  m.def("index_out(Tensor self, Tensor?[] indices, Tensor(a!) out) -> Tensor(a!)");
}
TORCH_LIBRARY_IMPL(hpc_index, CPU, m) {
  m.impl("index", &index_impl);
  m.impl("index_out", &index_out_impl);
}

// 直接从 C++ 覆盖 aten 的 CPU 内核：Python 侧 torch.library.impl 每个调用多一次
// Python 解释器往返（实测小 K 固定开销 ~10-20us），此处注册后 x[idx] 直达本内核。
namespace {
at::Tensor& index_out_aten(const at::Tensor& self,
                           const c10::List<c10::optional<at::Tensor>>& indices,
                           at::Tensor& out) {
  index_out_impl(self, indices, out);
  return out;
}
}  // namespace

TORCH_LIBRARY_IMPL(aten, CPU, m) {
  const char* e = getenv("HPC_INDEX_NO_OVERRIDE");
  if (e == nullptr || std::string(e) != "1") {
    m.impl("index.Tensor", &index_impl);
    m.impl("index.Tensor_out", &index_out_aten);
  }
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
  m.def("small_single_count", []() {
    return g_small_single.load(std::memory_order_relaxed);
  });
  m.def("small_multi_count", []() {
    return g_small_multi.load(std::memory_order_relaxed);
  });
  m.def("strided_rows_count", &strided_rows_count);
  m.def("rowcopy_raw_count", &rowcopy_raw_count);
  m.def("single_1d_count", []() {
    return g_single_1d.load(std::memory_order_relaxed);
  });
  m.def("single_1d_sve_count", []() {
    return g_single_1d_sve.load(std::memory_order_relaxed);
  });
  m.def("debug_ss_parallel", [](int64_t n, int64_t A) {
    return n * A >= small_single_par_threshold();
  });
  m.def("debug_grain", [](int64_t n) { return adaptive_grain(n); });
  m.def("idx_dtype_choice", [](int64_t numel, const std::string& dtype) {
    // 镜像 index_impl 的 use_i32 判定（仅供测试验证 f32 溢出回退逻辑）
    const bool is_4b = (dtype == "Float" || dtype == "Int");
    return (numel < (1LL << 31) && is_4b) ? 1 : 0;
  });
}
