#include "gather.h"
#include "grain.h"
#include <ATen/Parallel.h>
#include <algorithm>
#include <arm_sve.h>
#include <c10/util/irange.h>
#include <limits>
#include <mutex>
#include <cstring>
#include <vector>
#include <type_traits>
#include <atomic>

namespace {

constexpr int64_t PREFETCH_DIST = 16;  // 微基准扫描 {0,16,32,64} 后定的默认
std::atomic<uint64_t> g_strided_rows{0};
std::atomic<uint64_t> g_rowcopy_raw{0};

bool sve_512_ok() {
  // 仅当运行时 VL 恰为 512-bit 才启用 SVE 路径（编译按 -msve-vector-bits=512）
  static const bool ok = []() { return svcntb() == 64; }();
  return ok;
}

template <bool NT>
svfloat64_t gather8_f64(svbool_t pg, const double* base, svuint64_t off) {
  if constexpr (NT) return svldnt1_gather_u64index_f64(pg, base, off);
  return svld1_gather_u64index_f64(pg, base, off);
}
template <bool NT>
svint64_t gather8_s64(svbool_t pg, const int64_t* base, svuint64_t off) {
  if constexpr (NT) return svldnt1_gather_u64index_s64(pg, base, off);
  return svld1_gather_u64index_s64(pg, base, off);
}
template <bool NT>
void store8_f64(svbool_t pg, double* dst, svfloat64_t v) {
  if constexpr (NT) svstnt1_f64(pg, dst, v);
  else svst1_f64(pg, dst, v);
}
template <bool NT>
void store8_s64(svbool_t pg, int64_t* dst, svint64_t v) {
  if constexpr (NT) svstnt1_s64(pg, dst, v);
  else svst1_s64(pg, dst, v);
}
template <bool NT>
void store4_f32(svbool_t pg, float* dst, svfloat32_t v) {
  if constexpr (NT) svstnt1_f32(pg, dst, v);
  else svst1_f32(pg, dst, v);
}
template <bool NT>
void store4_s32(svbool_t pg, int32_t* dst, svint32_t v) {
  if constexpr (NT) svstnt1_s32(pg, dst, v);
  else svst1_s32(pg, dst, v);
}

// 8B 元素，idx 为 int64（元素偏移，u64index 按 esize 缩放）；主循环带 i+pf 预取
template <bool NT>
void pure_gather_sve_8b_f64(const double* src, const int64_t* idx, double* dst, int64_t n, int64_t pf) {
  const int64_t VL = svcntd();  // == 8 at VL512
  const uint64_t* idxu = reinterpret_cast<const uint64_t*>(idx);
  int64_t i = 0;
  if (pf > 0) {
    for (; i + pf + VL <= n; i += VL) {
      svbool_t pg = svptrue_b64();
      svuint64_t pf_off = svld1_u64(pg, idxu + i + pf);
      svprfd_gather_u64index(pg, src, pf_off, SV_PLDL1KEEP);
      svuint64_t offv = svld1_u64(pg, idxu + i);
      store8_f64<NT>(pg, dst + i, gather8_f64<NT>(pg, src, offv));
    }
  }
  for (; i + VL <= n; i += VL) {
    svbool_t pg = svptrue_b64();
    svuint64_t offv = svld1_u64(pg, idxu + i);
    store8_f64<NT>(pg, dst + i, gather8_f64<NT>(pg, src, offv));
  }
  if (i < n) {
    svbool_t pg = svwhilelt_b64(i, n);
    svuint64_t offv = svld1_u64(pg, idxu + i);
    store8_f64<NT>(pg, dst + i, gather8_f64<NT>(pg, src, offv));
  }
}

template <bool NT>
void pure_gather_sve_8b_s64(const int64_t* src, const int64_t* idx, int64_t* dst, int64_t n, int64_t pf) {
  const int64_t VL = svcntd();
  const uint64_t* idxu = reinterpret_cast<const uint64_t*>(idx);
  int64_t i = 0;
  if (pf > 0) {
    for (; i + pf + VL <= n; i += VL) {
      svbool_t pg = svptrue_b64();
      svuint64_t pf_off = svld1_u64(pg, idxu + i + pf);
      svprfd_gather_u64index(pg, src, pf_off, SV_PLDL1KEEP);
      svuint64_t offv = svld1_u64(pg, idxu + i);
      store8_s64<NT>(pg, dst + i, gather8_s64<NT>(pg, src, offv));
    }
  }
  for (; i + VL <= n; i += VL) {
    svbool_t pg = svptrue_b64();
    svuint64_t offv = svld1_u64(pg, idxu + i);
    store8_s64<NT>(pg, dst + i, gather8_s64<NT>(pg, src, offv));
  }
  if (i < n) {
    svbool_t pg = svwhilelt_b64(i, n);
    svuint64_t offv = svld1_u64(pg, idxu + i);
    store8_s64<NT>(pg, dst + i, gather8_s64<NT>(pg, src, offv));
  }
}

// 4B 元素，idx 为 int32（元素偏移，u32index 与 esize=4 匹配）；无 NT gather 变体，回退常规 load
template <bool NT>
void pure_gather_sve_4b_f32(const float* src, const int32_t* idx, float* dst, int64_t n, int64_t pf) {
  const int64_t VL = svcntw();  // == 16 at VL512
  const uint32_t* idxu = reinterpret_cast<const uint32_t*>(idx);
  int64_t i = 0;
  if (pf > 0) {
    for (; i + pf + VL <= n; i += VL) {
      svbool_t pg = svptrue_b32();
      svuint32_t pf_off = svld1_u32(pg, idxu + i + pf);
      svprfw_gather_u32index(pg, src, pf_off, SV_PLDL1KEEP);
      svuint32_t offv = svld1_u32(pg, idxu + i);
      store4_f32<NT>(pg, dst + i, svld1_gather_u32index_f32(pg, src, offv));
    }
  }
  for (; i + VL <= n; i += VL) {
    svbool_t pg = svptrue_b32();
    svuint32_t offv = svld1_u32(pg, idxu + i);
    store4_f32<NT>(pg, dst + i, svld1_gather_u32index_f32(pg, src, offv));
  }
  if (i < n) {
    svbool_t pg = svwhilelt_b32(i, n);
    svuint32_t offv = svld1_u32(pg, idxu + i);
    store4_f32<NT>(pg, dst + i, svld1_gather_u32index_f32(pg, src, offv));
  }
}

template <bool NT>
void pure_gather_sve_4b_s32(const int32_t* src, const int32_t* idx, int32_t* dst, int64_t n, int64_t pf) {
  const int64_t VL = svcntw();
  const uint32_t* idxu = reinterpret_cast<const uint32_t*>(idx);
  int64_t i = 0;
  if (pf > 0) {
    for (; i + pf + VL <= n; i += VL) {
      svbool_t pg = svptrue_b32();
      svuint32_t pf_off = svld1_u32(pg, idxu + i + pf);
      svprfw_gather_u32index(pg, src, pf_off, SV_PLDL1KEEP);
      svuint32_t offv = svld1_u32(pg, idxu + i);
      store4_s32<NT>(pg, dst + i, svld1_gather_u32index_s32(pg, src, offv));
    }
  }
  for (; i + VL <= n; i += VL) {
    svbool_t pg = svptrue_b32();
    svuint32_t offv = svld1_u32(pg, idxu + i);
    store4_s32<NT>(pg, dst + i, svld1_gather_u32index_s32(pg, src, offv));
  }
  if (i < n) {
    svbool_t pg = svwhilelt_b32(i, n);
    svuint32_t offv = svld1_u32(pg, idxu + i);
    store4_s32<NT>(pg, dst + i, svld1_gather_u32index_s32(pg, src, offv));
  }
}

// A==1 纯 gather：扁平输出 [0, B*K) 并行（B==1 时即按 K 分块，修复旧串行化）。
// 块内可能跨行，按行切段，段内连续 k 用一条向量 gather。
template <typename T, typename IdxT, bool NT>
void pure_gather_sve_flat(const IndexSpec& spec, const IdxT* idx, const T* src, T* dst, int64_t pf) {
  const int64_t B = spec.B, K = spec.K;
  const int64_t* row_off = spec.row_off.data();
  at::parallel_for(0, B * K, adaptive_grain(B * K), [&](int64_t lo, int64_t hi) {
    for (int64_t p = lo; p < hi;) {
      const int64_t r = p / K;
      const int64_t k = p - r * K;
      const int64_t end = std::min(hi, r * K + K);
      if constexpr (std::is_same_v<T, double>) {
        pure_gather_sve_8b_f64<NT>(src + row_off[r], idx + k, dst + p, end - p, pf);
      } else if constexpr (std::is_same_v<T, int64_t>) {
        pure_gather_sve_8b_s64<NT>(src + row_off[r], idx + k, dst + p, end - p, pf);
      } else if constexpr (std::is_same_v<T, float>) {
        pure_gather_sve_4b_f32<NT>(src + row_off[r], idx + k, dst + p, end - p, pf);
      } else {
        pure_gather_sve_4b_s32<NT>(src + row_off[r], idx + k, dst + p, end - p, pf);
      }
      p = end;
    }
  });
}

// 连续 A 元素块拷贝：8B svld1_f64/svst1_f64，4B svld1_f32/svst1_f32；4×VL 多寄存器
// 展开保持内存级并行，余下 VL 步进 + whilelt 尾数；NT 时 store 换 svstnt1_*
template <typename T, bool NT>
void copy_contig_sve(const T* src, T* dst, int64_t n) {
  if (n <= 0) return;
  // 小行宽（<=64B）用 memcpy，避免 masked SVE load/store 的谓词构造开销；
  // 非连续尾维的短 run（如 10×f32=40B）也走这里。
  if (n * static_cast<int64_t>(sizeof(T)) <= 64) {
    std::memcpy(dst, src, n * sizeof(T));
    return;
  }
  if constexpr (std::is_same_v<T, double>) {
    const int64_t VL = svcntd();
    int64_t i = 0;
    for (; i + 4 * VL <= n; i += 4 * VL) {
      svfloat64_t v0 = svld1_f64(svptrue_b64(), src + i);
      svfloat64_t v1 = svld1_f64(svptrue_b64(), src + i + VL);
      svfloat64_t v2 = svld1_f64(svptrue_b64(), src + i + 2 * VL);
      svfloat64_t v3 = svld1_f64(svptrue_b64(), src + i + 3 * VL);
      store8_f64<NT>(svptrue_b64(), dst + i, v0);
      store8_f64<NT>(svptrue_b64(), dst + i + VL, v1);
      store8_f64<NT>(svptrue_b64(), dst + i + 2 * VL, v2);
      store8_f64<NT>(svptrue_b64(), dst + i + 3 * VL, v3);
    }
    for (; i + VL <= n; i += VL) {
      svfloat64_t v = svld1_f64(svptrue_b64(), src + i);
      store8_f64<NT>(svptrue_b64(), dst + i, v);
    }
    if (i < n) {
      svbool_t pg = svwhilelt_b64(i, n);
      store8_f64<NT>(pg, dst + i, svld1_f64(pg, src + i));
    }
  } else if constexpr (std::is_same_v<T, float>) {
    const int64_t VL = svcntw();
    int64_t i = 0;
    for (; i + 4 * VL <= n; i += 4 * VL) {
      svfloat32_t v0 = svld1_f32(svptrue_b32(), src + i);
      svfloat32_t v1 = svld1_f32(svptrue_b32(), src + i + VL);
      svfloat32_t v2 = svld1_f32(svptrue_b32(), src + i + 2 * VL);
      svfloat32_t v3 = svld1_f32(svptrue_b32(), src + i + 3 * VL);
      store4_f32<NT>(svptrue_b32(), dst + i, v0);
      store4_f32<NT>(svptrue_b32(), dst + i + VL, v1);
      store4_f32<NT>(svptrue_b32(), dst + i + 2 * VL, v2);
      store4_f32<NT>(svptrue_b32(), dst + i + 3 * VL, v3);
    }
    for (; i + VL <= n; i += VL) {
      svfloat32_t v = svld1_f32(svptrue_b32(), src + i);
      store4_f32<NT>(svptrue_b32(), dst + i, v);
    }
    if (i < n) {
      svbool_t pg = svwhilelt_b32(i, n);
      store4_f32<NT>(pg, dst + i, svld1_f32(pg, src + i));
    }
  }
}

struct ColRun {
  int64_t dst;
  int64_t src;
  int64_t len;
};

// 把 col_off 中相邻（+1）的元素合并成连续 run。尾维非连续时，每个 run 可一次
// 块拷贝；run 长度=1 表示纯跨步，退化为逐元素。
std::vector<ColRun> make_col_runs(const int64_t* col_off, int64_t A) {
  std::vector<ColRun> runs;
  int64_t c = 0;
  while (c < A) {
    int64_t len = 1;
    while (c + len < A && col_off[c + len] == col_off[c + len - 1] + 1) ++len;
    runs.push_back({c, col_off[c], len});
    c += len;
  }
  return runs;
}

// 单索引、after_contiguous=false：按连续 run 做块拷贝（fastindex run_strided_rows
// 的等价实现）。run=1 不优于 fused，由调用方保留原路径。
template <typename T, typename IdxT, bool NT>
void strided_rows_raw(const IndexSpec& spec, const IdxT* raw, T* dst, const T* src,
                      const std::vector<ColRun>& runs) {
  const int64_t B = spec.B, K = spec.K, A = spec.A;
  const int64_t size = spec.indexed_sizes[0];
  const int64_t stride = spec.indexed_strides[0];
  const int64_t* row_off = spec.row_off.data();
  at::parallel_for(0, B * K, adaptive_grain(B * K), [&](int64_t lo, int64_t hi) {
    for (int64_t p = lo; p < hi; ++p) {
      const int64_t r = p / K, k = p % K;
      int64_t v = static_cast<int64_t>(raw[k]);
      TORCH_CHECK_INDEX(v >= -size && v < size,
          "index ", v, " is out of bounds for dimension 0 with size ", size);
      if (v < 0) v += size;
      const T* s = src + row_off[r] + v * stride;
      T* d = dst + p * A;
      for (const auto& run : runs) {
        copy_contig_sve<T, NT>(s + run.src, d + run.dst, run.len);
      }
    }
  });
}

// A>1 && after_contiguous：整行连续块拷贝 out[(r*K+k)*A+c] = src[row_off[r] + idx[k] + c]
template <typename T, typename IdxT, bool NT>
void rowcopy_sve(const T* src, const IdxT* idx, T* dst,
                 const int64_t* row_off, int64_t B, int64_t K, int64_t A) {
  at::parallel_for(0, B * K, adaptive_grain(B * K), [&](int64_t lo, int64_t hi) {
    for (int64_t p = lo; p < hi; ++p) {
      const int64_t r = p / K, k = p % K;
      const T* s = src + row_off[r] + idx[k];
      T* d = dst + p * A;
      copy_contig_sve<T, NT>(s, d, A);
    }
  });
}

template <typename T, typename IdxT>
void scalar_gather(const T* src, const IdxT* idx, T* dst,
                   const int64_t* row_off, const int64_t* col_off,
                   int64_t B, int64_t K, int64_t A) {
  at::parallel_for(0, B * K, adaptive_grain(B * K), [&](int64_t lo, int64_t hi) {
    for (int64_t p = lo; p < hi; ++p) {
      const int64_t r = p / K, k = p % K;
      const T* s = src + row_off[r] + idx[k];
      T* d = dst + p * A;
      for (const auto c : c10::irange(A)) {
        d[c] = s[col_off[c]];
      }
    }
  });
}

template <typename T, typename IdxT>
void dispatch_scalar(const IndexSpec& spec, const IdxT* idx, T* out_data, const T* src) {
  scalar_gather(src, idx, out_data, spec.row_off.data(), spec.col_off.data(),
                spec.B, spec.K, spec.A);
}

// T4 小值域短路（A==1 纯 gather）：linearize 后先求偏移值域，≤2048 时用紧凑查找表 +
// 标量循环。offset 表小、源数据块常驻 L1/L2，标量查表命中缓存；4B SVE gather 受
// 64B 行放大限制（RESULTS §8.4），此形态查表更快。返回是否启用（值域过大时留给 SVE）。
template <typename T, typename IdxT>
bool bucket_gather_pure(const IndexSpec& spec, const IdxT* idx, T* dst, const T* src) {
  const int64_t B = spec.B, K = spec.K;
  const int64_t n = B * K;
  constexpr int64_t MAX_DOMAIN = 2048;  // 8B 表 16KB / 4B 表 8KB，均 L1/L2 常驻
  const int64_t* row_off = spec.row_off.data();
  const int64_t grain = adaptive_grain(n);
  const int64_t nchunk = (n + grain - 1) / grain;
  std::vector<int64_t> vmn(nchunk), vmx(nchunk);
  at::parallel_for(0, n, grain, [&](int64_t lo, int64_t hi) {
    const int64_t ci = lo / grain;
    int64_t lmn = std::numeric_limits<int64_t>::max();
    int64_t lmx = std::numeric_limits<int64_t>::min();
    for (int64_t p = lo; p < hi; ++p) {
      const int64_t v = static_cast<int64_t>(row_off[p / K] + idx[p - (p / K) * K]);
      lmn = std::min(lmn, v);
      lmx = std::max(lmx, v);
    }
    vmn[ci] = lmn;
    vmx[ci] = lmx;
  });
  int64_t mn = std::numeric_limits<int64_t>::max();
  int64_t mx = std::numeric_limits<int64_t>::min();
  for (const auto i : c10::irange(nchunk)) {
    mn = std::min(mn, vmn[i]);
    mx = std::max(mx, vmx[i]);
  }
  const int64_t dom = mx - mn + 1;
  if (dom <= 1 || dom > MAX_DOMAIN) return false;
  std::vector<T> table(dom);
  for (int64_t v = 0; v < dom; ++v) table[v] = src[mn + v];
  at::parallel_for(0, n, grain, [&](int64_t lo, int64_t hi) {
    for (int64_t p = lo; p < hi; ++p) {
      const int64_t v = static_cast<int64_t>(row_off[p / K] + idx[p - (p / K) * K]) - mn;
      dst[p] = table[v];
    }
  });
  return true;
}
// T4 raw 变体：单索引 + B==1 纯 gather（M3 keys[order]、M4 等 1D 形态）完全跳过 linearize，
// 直接在原始下标上算偏移求值域。B==1 时 offset = idx[k]（row_off=col_off=[0]）。
template <typename T, typename IdxT>
bool bucket_gather_pure_raw(const IndexSpec& spec, const IdxT* idx, T* dst, const T* src) {
  const int64_t n = spec.K;
  const int64_t stride0 = spec.indexed_strides[0];
  constexpr int64_t MAX_DOMAIN = 2048;
  const int64_t grain = adaptive_grain(n);
  const int64_t nchunk = (n + grain - 1) / grain;
  std::vector<int64_t> vmn(nchunk), vmx(nchunk);
  at::parallel_for(0, n, grain, [&](int64_t lo, int64_t hi) {
    const int64_t ci = lo / grain;
    int64_t lmn = std::numeric_limits<int64_t>::max();
    int64_t lmx = std::numeric_limits<int64_t>::min();
    for (int64_t k = lo; k < hi; ++k) {
      int64_t v = static_cast<int64_t>(idx[k]);
      TORCH_CHECK_INDEX(v >= -spec.indexed_sizes[0] && v < spec.indexed_sizes[0],
          "index ", v, " is out of bounds for dimension 0 with size ", spec.indexed_sizes[0]);
      if (v < 0) v += spec.indexed_sizes[0];
      lmn = std::min(lmn, v);
      lmx = std::max(lmx, v);
    }
    vmn[ci] = lmn;
    vmx[ci] = lmx;
  });
  int64_t mn = std::numeric_limits<int64_t>::max();
  int64_t mx = std::numeric_limits<int64_t>::min();
  for (const auto i : c10::irange(nchunk)) {
    mn = std::min(mn, vmn[i]);
    mx = std::max(mx, vmx[i]);
  }
  const int64_t dom = mx - mn + 1;
  if (dom <= 1 || dom > MAX_DOMAIN) return false;
  std::vector<T> table(dom);
  for (int64_t v = 0; v < dom; ++v) table[v] = src[(mn + v) * stride0];
  at::parallel_for(0, n, grain, [&](int64_t lo, int64_t hi) {
    for (int64_t k = lo; k < hi; ++k) {
      int64_t v = static_cast<int64_t>(idx[k]);
      if (v < 0) v += spec.indexed_sizes[0];
      dst[k] = table[v - mn];
    }
  });
  return true;
}

// K<1024 快速路径（spec §6 阈值）：跳过线性化，直接读原始下标 + 内联检查（等价 ATen Indexer）
template <typename T>
void gather_direct(const IndexSpec& spec, T* dst, const T* src) {
  const int64_t nidx = spec.indexed_dims.size();
  const int64_t B = spec.B, K = spec.K, A = spec.A;
  const std::vector<int64_t>& bshape = spec.broadcast_shape;
  const int64_t nb = bshape.size();
  std::vector<int64_t> bfactor(nb, 1);          // 行优先混合进制因子（A7）
  for (int64_t i = nb - 2; i >= 0; --i) {
    bfactor[i] = bfactor[i + 1] * bshape[i + 1];
  }
  // 每元素调用 at::Tensor 方法（element_size/stride/data_ptr/dtype）在 B*K 大时
  // 是主要开销（x[:,i] 实测 1t 0.18x）；循环外提取一次。
  std::vector<const void*> ind_ptr(nidx);
  std::vector<const int64_t*> ind_stride(nidx);
  std::vector<int64_t> ind_size(nidx), ind_esize(nidx);
  std::vector<bool> ind_long(nidx);
  bool all_full_contig = (nb == 1);
  for (const auto j : c10::irange(nidx)) {
    const at::Tensor& ind = spec.indices[j];
    ind_ptr[j] = ind.data_ptr();
    ind_stride[j] = ind.strides().data();
    ind_size[j] = spec.indexed_sizes[j];
    ind_esize[j] = ind.element_size();
    ind_long[j] = (ind.dtype() == at::kLong);
    if (ind.numel() != K || !ind.is_contiguous()) all_full_contig = false;
  }
  // A==1 + 单索引 + 全长连续：先把 K 个下标校验/归一化一次，再跑无检查的紧凑循环。
  // 逐元素边界检查 + col_off 内层循环在 x[:,i]（B 大、K 小、A=1）上占主要开销，
  // 提到循环外后内层等价 aten 的 dst[i]=src[idx[i]]（见 degradation-analysis.md）。
  if (A == 1 && nidx == 1 && all_full_contig && spec.after_contiguous) {
    const int64_t size = ind_size[0];
    const int64_t stride0 = spec.indexed_strides[0];
    const int64_t col0 = spec.col_off.empty() ? 0 : spec.col_off[0];
    const int32_t* i32 = ind_long[0] ? nullptr
        : static_cast<const int32_t*>(ind_ptr[0]);
    const int64_t* i64 = ind_long[0]
        ? static_cast<const int64_t*>(ind_ptr[0]) : nullptr;
    std::vector<int64_t> vbuf(K);
    for (const auto k : c10::irange(K)) {
      int64_t v = i64 ? i64[k] : static_cast<int64_t>(i32[k]);
      TORCH_CHECK_INDEX(v >= -size && v < size,
          "index ", v, " is out of bounds for dimension 0 with size ", size);
      vbuf[k] = v < 0 ? v + size : v;
    }
    const int64_t* row_off = spec.row_off.data();
    const int64_t* vp = vbuf.data();
    at::parallel_for(0, B * K, adaptive_grain(B * K), [&](int64_t lo, int64_t hi) {
      int64_t r = lo / K, k = lo % K;
      for (int64_t p = lo; p < hi; ++p) {
        dst[p] = src[row_off[r] + vp[k] * stride0 + col0];
        if (++k == K) { k = 0; ++r; }
      }
    });
    return;
  }
  at::parallel_for(0, B * K, adaptive_grain(B * K), [&](int64_t lo, int64_t hi) {
    int64_t r = lo / K, k = lo % K;
    for (int64_t p = lo; p < hi; ++p) {
      int64_t off = spec.row_off[r];
      for (const auto j : c10::irange(nidx)) {
        const char* ptr;
        if (all_full_contig) {
          ptr = static_cast<const char*>(ind_ptr[j]) + k * ind_esize[j];
        } else {
          int64_t eoff = 0, rr = k;
          for (const auto d : c10::irange(nb)) {   // A7：按 broadcast_shape 坐标读元素
            int64_t c = (bshape[d] > 0) ? (rr / bfactor[d]) % bshape[d] : 0;
            eoff += c * ind_stride[j][d];
            if (bshape[d] > 0) rr %= bfactor[d];
          }
          ptr = static_cast<const char*>(ind_ptr[j]) + eoff * ind_esize[j];
        }
        int64_t v = ind_long[j]
            ? *reinterpret_cast<const int64_t*>(ptr)
            : *reinterpret_cast<const int32_t*>(ptr);
        const int64_t size = ind_size[j];
        TORCH_CHECK_INDEX(v >= -size && v < size,
            "index ", v, " is out of bounds for dimension ", j, " with size ", size);
        if (v < 0) v += size;
        off += v * spec.indexed_strides[j];
      }
      T* d = dst + p * A;
      const T* s = src + off;
      for (const auto c : c10::irange(A)) {
        d[c] = s[spec.col_off[c]];
      }
      if (++k == K) { k = 0; ++r; }
    }
  });
}

// T3：linearize+gather 融合单遍（scalar 小块路径：A<4 或非连续）。与 gather_direct
// 同源：并行循环内先算偏移（内联边界检查/负索引规整），再就地标量拷贝；不物化 idx
// 数组（P2）。仅启用于此路径——A>=4 连续块实测 linearize+rowcopy 更快；A==1 纯
// gather 需 idx 常驻内存走 SVE。与 T2 分支互斥（T2 只处理单索引大块）。
template <typename T>
void gather_fused(const IndexSpec& spec, T* dst, const T* src) {
  const int64_t nidx = spec.indexed_dims.size();
  const int64_t B = spec.B, K = spec.K, A = spec.A;
  const std::vector<int64_t>& bshape = spec.broadcast_shape;
  const int64_t nb = bshape.size();
  std::vector<int64_t> bfactor(nb, 1);
  for (int64_t i = nb - 2; i >= 0; --i) {
    bfactor[i] = bfactor[i + 1] * bshape[i + 1];
  }
  // 预提取各下标张量的指针/stride/size/esize/长整型标志，避免内层循环每次访问
  // at::Tensor 方法（引用计数 + 虚调用），这是 fused 单遍慢于 aten 的根因。
  std::vector<const void*> ind_ptr(nidx);
  std::vector<const int64_t*> ind_stride(nidx);
  std::vector<int64_t> ind_size(nidx);
  std::vector<int64_t> ind_esize(nidx);
  std::vector<bool> ind_long(nidx);
  bool all_full_contig = (nb == 1);
  for (const auto j : c10::irange(nidx)) {
    const at::Tensor& ind = spec.indices[j];
    ind_ptr[j] = ind.data_ptr();
    ind_stride[j] = ind.strides().data();
    ind_size[j] = spec.indexed_sizes[j];
    ind_esize[j] = ind.element_size();
    ind_long[j] = (ind.dtype() == at::kLong);
    if (ind.numel() != K || !ind.is_contiguous()) all_full_contig = false;
  }
  // nb==1 快速路径：所有索引与 K 全长对齐且连续（无广播展开，如 x[i,j] 双全长索引、
  // 单索引 rowcopy），解码退化为直读 idx_j[k]，省掉 bfactor 混合进制展开。
  if (all_full_contig) {
    at::parallel_for(0, B * K, adaptive_grain(B * K), [&](int64_t lo, int64_t hi) {
      for (int64_t p = lo; p < hi; ++p) {
        const int64_t r = p / K, k = p % K;
        int64_t off = spec.row_off[r];
        for (const auto j : c10::irange(nidx)) {
          const char* ptr = static_cast<const char*>(ind_ptr[j]) + k * ind_esize[j];
          int64_t v = ind_long[j]
              ? *reinterpret_cast<const int64_t*>(ptr)
              : *reinterpret_cast<const int32_t*>(ptr);
          const int64_t size = ind_size[j];
          TORCH_CHECK_INDEX(v >= -size && v < size,
              "index ", v, " is out of bounds for dimension ", j, " with size ", size);
          if (v < 0) v += size;
          off += v * spec.indexed_strides[j];
        }
        T* d = dst + p * A;
        const T* s = src + off;
        for (const auto c : c10::irange(A)) {
          d[c] = s[spec.col_off[c]];
        }
      }
    });
    return;
  }
  at::parallel_for(0, B * K, adaptive_grain(B * K), [&](int64_t lo, int64_t hi) {
    for (int64_t p = lo; p < hi; ++p) {
      const int64_t r = p / K, k = p % K;
      int64_t off = spec.row_off[r];
      for (const auto j : c10::irange(nidx)) {
        const int64_t esize = ind_esize[j];
        const int64_t* ind_st = ind_stride[j];
        int64_t eoff = 0;
        int64_t rr = k;
        for (const auto d : c10::irange(nb)) {
          int64_t c = (bshape[d] > 0) ? (rr / bfactor[d]) % bshape[d] : 0;
          eoff += c * ind_st[d];
          if (bshape[d] > 0) rr %= bfactor[d];
        }
        const char* ptr = static_cast<const char*>(ind_ptr[j]) + eoff * esize;
        int64_t v = ind_long[j]
            ? *reinterpret_cast<const int64_t*>(ptr)
            : *reinterpret_cast<const int32_t*>(ptr);
        const int64_t size = ind_size[j];
        TORCH_CHECK_INDEX(v >= -size && v < size,
            "index ", v, " is out of bounds for dimension ", j, " with size ", size);
        if (v < 0) v += size;
        off += v * spec.indexed_strides[j];
      }
      T* d = dst + p * A;
      const T* s = src + off;
      for (const auto c : c10::irange(A)) {
        d[c] = s[spec.col_off[c]];
      }
    }
  });
}

}  // namespace

void gather_index(const IndexSpec& spec, const at::Tensor& idx, at::Tensor& out,
                  bool non_temporal) {
  const int64_t* row_off = spec.row_off.data();
  const int64_t* col_off = spec.col_off.data();
  const int64_t B = spec.B, K = spec.K, A = spec.A;
  const bool idx32 = idx.scalar_type() == at::kInt;
  const bool pure = (A == 1);
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(at::kHalf, at::kBFloat16, at::kBool,
      out.scalar_type(), "hpc_index_gather", [&] {
    const scalar_t* src = static_cast<const scalar_t*>(spec.transposed_self.data_ptr());
    scalar_t* dst = static_cast<scalar_t*>(out.data_ptr());
    if (pure && idx32) {
      if (bucket_gather_pure<scalar_t, int32_t>(spec, static_cast<const int32_t*>(idx.data_ptr()), dst, src)) return;
    } else if (pure) {
      if (bucket_gather_pure<scalar_t, int64_t>(spec, static_cast<const int64_t*>(idx.data_ptr()), dst, src)) return;
    }
    if (pure && !idx32 && sve_512_ok() &&
        (out.scalar_type() == at::kDouble || out.scalar_type() == at::kLong)) {
      const int64_t* idx64 = static_cast<const int64_t*>(idx.data_ptr());
      if (non_temporal) {
        if (out.scalar_type() == at::kDouble) {
          pure_gather_sve_flat<double, int64_t, true>(
              spec, idx64, reinterpret_cast<const double*>(src), reinterpret_cast<double*>(dst), PREFETCH_DIST);
        } else {
          pure_gather_sve_flat<int64_t, int64_t, true>(
              spec, idx64, reinterpret_cast<const int64_t*>(src), reinterpret_cast<int64_t*>(dst), PREFETCH_DIST);
        }
      } else {
        if (out.scalar_type() == at::kDouble) {
          pure_gather_sve_flat<double, int64_t, false>(
              spec, idx64, reinterpret_cast<const double*>(src), reinterpret_cast<double*>(dst), PREFETCH_DIST);
        } else {
          pure_gather_sve_flat<int64_t, int64_t, false>(
              spec, idx64, reinterpret_cast<const int64_t*>(src), reinterpret_cast<int64_t*>(dst), PREFETCH_DIST);
        }
      }
      return;
    }
    if (pure && idx32 && sve_512_ok() &&
        (out.scalar_type() == at::kFloat || out.scalar_type() == at::kInt)) {
      const int32_t* idx32p = static_cast<const int32_t*>(idx.data_ptr());
      if (non_temporal) {
        if (out.scalar_type() == at::kFloat) {
          pure_gather_sve_flat<float, int32_t, true>(
              spec, idx32p, reinterpret_cast<const float*>(src), reinterpret_cast<float*>(dst), PREFETCH_DIST);
        } else {
          pure_gather_sve_flat<int32_t, int32_t, true>(
              spec, idx32p, reinterpret_cast<const int32_t*>(src), reinterpret_cast<int32_t*>(dst), PREFETCH_DIST);
        }
      } else {
        if (out.scalar_type() == at::kFloat) {
          pure_gather_sve_flat<float, int32_t, false>(
              spec, idx32p, reinterpret_cast<const float*>(src), reinterpret_cast<float*>(dst), PREFETCH_DIST);
        } else {
          pure_gather_sve_flat<int32_t, int32_t, false>(
              spec, idx32p, reinterpret_cast<const int32_t*>(src), reinterpret_cast<int32_t*>(dst), PREFETCH_DIST);
        }
      }
      return;
    }
    const bool rowcopy = (A > 1);
    if (rowcopy && spec.after_contiguous && sve_512_ok() && A >= 4 &&
        (out.scalar_type() == at::kFloat || out.scalar_type() == at::kDouble)) {
      if (non_temporal) {
        if (idx32) {
          rowcopy_sve<scalar_t, int32_t, true>(src, static_cast<const int32_t*>(idx.data_ptr()),
                                               dst, row_off, B, K, A);
        } else {
          rowcopy_sve<scalar_t, int64_t, true>(src, static_cast<const int64_t*>(idx.data_ptr()),
                                               dst, row_off, B, K, A);
        }
      } else {
        if (idx32) {
          rowcopy_sve<scalar_t, int32_t, false>(src, static_cast<const int32_t*>(idx.data_ptr()),
                                                dst, row_off, B, K, A);
        } else {
          rowcopy_sve<scalar_t, int64_t, false>(src, static_cast<const int64_t*>(idx.data_ptr()),
                                                dst, row_off, B, K, A);
        }
      }
      return;
    }
    (void)col_off;
    if (idx32) {
      dispatch_scalar(spec, static_cast<const int32_t*>(idx.data_ptr()), dst, src);
    } else {
      dispatch_scalar(spec, static_cast<const int64_t*>(idx.data_ptr()), dst, src);
    }
  });
}

void gather_direct(const IndexSpec& spec, at::Tensor& out) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(at::kHalf, at::kBFloat16, at::kBool,
      out.scalar_type(), "hpc_index_gather_direct", [&] {
    gather_direct(spec, static_cast<scalar_t*>(out.data_ptr()),
                  static_cast<const scalar_t*>(spec.transposed_self.data_ptr()));
  });
}

// 1D 单索引纯 gather 专用单遍内核（1 线程路径）：跳过 linearize，把下标张量的
// data_ptr/stride/size/dtype 提到循环外一次提取，循环内等价 aten 单遍标量
// （read idx → 边界检查 → 负索引归一 → out[k]=src[off]）。SVE gather 单线程在此
// 形态不占优（4B 受 64B 行放大限制，8B 收益也有限），linearize 两遍又是固定开销，
// 1 线程用本内核追平 aten。
template <typename T, typename IdxT>
void gather_single_1d(const IndexSpec& spec, const IdxT* idx, T* dst, const T* src) {
  const int64_t n = spec.K;
  const int64_t size = spec.indexed_sizes[0];
  const int64_t stride0 = spec.indexed_strides[0];
  at::parallel_for(0, n, adaptive_grain(n), [&](int64_t lo, int64_t hi) {
    for (int64_t k = lo; k < hi; ++k) {
      int64_t v = static_cast<int64_t>(idx[k]);
      TORCH_CHECK_INDEX(v >= -size && v < size,
          "index ", v, " is out of bounds for dimension 0 with size ", size);
      if (v < 0) v += size;
      dst[k] = src[v * stride0];
    }
  });
}

// T4 raw 包装：单索引 1D 纯 gather 时跳过 linearize（省掉 broadcast 解码），直接在原始
// 下标上求值域并查表。调用方需保证 spec.indices.size()==1 && B==1 && A==1（M3/M4 形态）。
bool gather_bucket_pure_raw(const IndexSpec& spec, at::Tensor& out) {
  const at::Tensor& ind = spec.indices[0];
  const bool idx32 = ind.scalar_type() == at::kInt;
  bool used = false;
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(at::kHalf, at::kBFloat16, at::kBool,
      out.scalar_type(), "hpc_index_bucket_pure_raw", [&] {
    const scalar_t* src = static_cast<const scalar_t*>(spec.transposed_self.data_ptr());
    scalar_t* dst = static_cast<scalar_t*>(out.data_ptr());
    if (idx32) {
      used = bucket_gather_pure_raw<scalar_t, int32_t>(spec, static_cast<const int32_t*>(ind.data_ptr()), dst, src);
    } else {
      used = bucket_gather_pure_raw<scalar_t, int64_t>(spec, static_cast<const int64_t*>(ind.data_ptr()), dst, src);
    }
  });
  return used;
}

// 1D 单索引纯 gather 单遍（1 线程路径）：跳过 linearize + SVE gather，直接标量单遍。
void gather_single_1d(const IndexSpec& spec, at::Tensor& out) {
  const at::Tensor& ind = spec.indices[0];
  const bool idx32 = ind.scalar_type() == at::kInt;
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(at::kHalf, at::kBFloat16, at::kBool,
      out.scalar_type(), "hpc_index_gather_single_1d", [&] {
    const scalar_t* src = static_cast<const scalar_t*>(spec.transposed_self.data_ptr());
    scalar_t* dst = static_cast<scalar_t*>(out.data_ptr());
    if (idx32) {
      gather_single_1d<scalar_t, int32_t>(spec, static_cast<const int32_t*>(ind.data_ptr()), dst, src);
    } else {
      gather_single_1d<scalar_t, int64_t>(spec, static_cast<const int64_t*>(ind.data_ptr()), dst, src);
    }
  });
}

// M1 单索引大块 skip-linearize：不物化 idx，直接在 gather 阶段读原始下标、内联边界
// 检查与负索引规整（等价 linearize 内循环），随后对 A 连续块做 SVE rowcopy。只有
// spec.after_contiguous 才启用。out 行序 = 原下标序（只改变 src 读序由 T2 阶段 2 处理）。
template <typename T, typename IdxT, bool NT>
void rowcopy_raw_single(const IndexSpec& spec, const IdxT* raw, T* dst, const T* src) {
  const int64_t B = spec.B, K = spec.K, A = spec.A;
  const int64_t size = spec.indexed_sizes[0];
  const int64_t stride = spec.indexed_strides[0];
  const int64_t* row_off = spec.row_off.data();
  auto row = [&](int64_t p, int64_t r, int64_t k) {
    int64_t v = static_cast<int64_t>(raw[k]);
    TORCH_CHECK_INDEX(v >= -size && v < size,
        "index ", v, " is out of bounds for dimension 0 with size ", size);
    if (v < 0) v += size;
    copy_contig_sve<T, NT>(src + row_off[r] + v * stride, dst + p * A, A);
  };
  if (B == 1) {
    at::parallel_for(0, K, adaptive_grain(K), [&](int64_t lo, int64_t hi) {
      for (int64_t k = lo; k < hi; ++k) row(k, 0, k);
    });
    return;
  }
  at::parallel_for(0, B * K, adaptive_grain(B * K), [&](int64_t lo, int64_t hi) {
    for (int64_t p = lo; p < hi; ++p) row(p, p / K, p % K);
  });
}

void gather_rowcopy_raw(const IndexSpec& spec, at::Tensor& out, bool non_temporal) {
  g_rowcopy_raw.fetch_add(1, std::memory_order_relaxed);
  const at::Tensor& raw = spec.indices[0];
  const bool idx32 = raw.scalar_type() == at::kInt;
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(at::kHalf, at::kBFloat16, at::kBool,
      out.scalar_type(), "hpc_index_rowcopy_raw", [&] {
    const scalar_t* src = static_cast<const scalar_t*>(spec.transposed_self.data_ptr());
    scalar_t* dst = static_cast<scalar_t*>(out.data_ptr());
    if (non_temporal) {
      if (idx32) {
        rowcopy_raw_single<scalar_t, int32_t, true>(
            spec, static_cast<const int32_t*>(raw.data_ptr()), dst, src);
      } else {
        rowcopy_raw_single<scalar_t, int64_t, true>(
            spec, static_cast<const int64_t*>(raw.data_ptr()), dst, src);
      }
    } else {
      if (idx32) {
        rowcopy_raw_single<scalar_t, int32_t, false>(
            spec, static_cast<const int32_t*>(raw.data_ptr()), dst, src);
      } else {
        rowcopy_raw_single<scalar_t, int64_t, false>(
            spec, static_cast<const int64_t*>(raw.data_ptr()), dst, src);
      }
    }
  });
}

template <typename T, bool NT>
void rowcopy_fused_sve(const IndexSpec& spec, T* dst, const T* src) {
  const int64_t nidx = spec.indexed_dims.size();
  const int64_t B = spec.B, K = spec.K, A = spec.A;
  const std::vector<int64_t>& bshape = spec.broadcast_shape;
  const int64_t nb = bshape.size();
  std::vector<int64_t> bfactor(nb, 1);
  for (int64_t i = nb - 2; i >= 0; --i) bfactor[i] = bfactor[i + 1] * bshape[i + 1];
  std::vector<const void*> ind_ptr(nidx);
  std::vector<const int64_t*> ind_stride(nidx);
  std::vector<int64_t> ind_size(nidx), ind_esize(nidx);
  std::vector<bool> ind_long(nidx);
  bool all_full_contig = (nb == 1);
  for (const auto j : c10::irange(nidx)) {
    const at::Tensor& ind = spec.indices[j];
    ind_ptr[j] = ind.data_ptr();
    ind_stride[j] = ind.strides().data();
    ind_size[j] = spec.indexed_sizes[j];
    ind_esize[j] = ind.element_size();
    ind_long[j] = (ind.dtype() == at::kLong);
    if (ind.numel() != K || !ind.is_contiguous()) all_full_contig = false;
  }
  if (all_full_contig) {
    at::parallel_for(0, B * K, adaptive_grain(B * K), [&](int64_t lo, int64_t hi) {
      for (int64_t p = lo; p < hi; ++p) {
        const int64_t r = p / K, k = p % K;
        int64_t off = spec.row_off[r];
        for (const auto j : c10::irange(nidx)) {
          const char* ptr = static_cast<const char*>(ind_ptr[j]) + k * ind_esize[j];
          int64_t v = ind_long[j]
              ? *reinterpret_cast<const int64_t*>(ptr)
              : *reinterpret_cast<const int32_t*>(ptr);
          const int64_t size = ind_size[j];
          TORCH_CHECK_INDEX(v >= -size && v < size,
              "index ", v, " is out of bounds for dimension ", j, " with size ", size);
          if (v < 0) v += size;
          off += v * spec.indexed_strides[j];
        }
        copy_contig_sve<T, NT>(src + off, dst + p * A, A);
      }
    });
    return;
  }
  at::parallel_for(0, B * K, adaptive_grain(B * K), [&](int64_t lo, int64_t hi) {
    for (int64_t p = lo; p < hi; ++p) {
      const int64_t r = p / K, k = p % K;
      int64_t off = spec.row_off[r];
      for (const auto j : c10::irange(nidx)) {
        const int64_t esize = ind_esize[j];
        const int64_t* ind_st = ind_stride[j];
        int64_t eoff = 0, rr = k;
        for (const auto d : c10::irange(nb)) {
          int64_t c = (bshape[d] > 0) ? (rr / bfactor[d]) % bshape[d] : 0;
          eoff += c * ind_st[d];
          if (bshape[d] > 0) rr %= bfactor[d];
        }
        const char* ptr = static_cast<const char*>(ind_ptr[j]) + eoff * esize;
        int64_t v = ind_long[j]
            ? *reinterpret_cast<const int64_t*>(ptr)
            : *reinterpret_cast<const int32_t*>(ptr);
        const int64_t size = ind_size[j];
        TORCH_CHECK_INDEX(v >= -size && v < size,
            "index ", v, " is out of bounds for dimension ", j, " with size ", size);
        if (v < 0) v += size;
        off += v * spec.indexed_strides[j];
      }
      copy_contig_sve<T, NT>(src + off, dst + p * A, A);
    }
  });
}

void gather_rowcopy_fused(const IndexSpec& spec, at::Tensor& out, bool non_temporal) {
  g_rowcopy_raw.fetch_add(1, std::memory_order_relaxed);
  const auto st = out.scalar_type();
  if (st == at::kFloat) {
    const float* src = static_cast<const float*>(spec.transposed_self.data_ptr());
    float* dst = static_cast<float*>(out.data_ptr());
    if (non_temporal) rowcopy_fused_sve<float, true>(spec, dst, src);
    else rowcopy_fused_sve<float, false>(spec, dst, src);
  } else {
    const double* src = static_cast<const double*>(spec.transposed_self.data_ptr());
    double* dst = static_cast<double*>(out.data_ptr());
    if (non_temporal) rowcopy_fused_sve<double, true>(spec, dst, src);
    else rowcopy_fused_sve<double, false>(spec, dst, src);
  }
}

bool gather_strided_rows_raw(const IndexSpec& spec, at::Tensor& out, bool non_temporal) {
  const int64_t A = spec.A;
  auto runs = make_col_runs(spec.col_off.data(), A);
  int64_t max_run = 0;
  for (const auto& run : runs) max_run = std::max(max_run, run.len);
  if (max_run < 4) return false;
  const auto st = out.scalar_type();
  if (st != at::kFloat && st != at::kDouble) return false;
  g_strided_rows.fetch_add(1, std::memory_order_relaxed);
  const at::Tensor& raw = spec.indices[0];
  const bool idx32 = raw.scalar_type() == at::kInt;
  if (st == at::kFloat) {
    const float* src = static_cast<const float*>(spec.transposed_self.data_ptr());
    float* dst = static_cast<float*>(out.data_ptr());
    if (idx32) {
      if (non_temporal) strided_rows_raw<float, int32_t, true>(spec, static_cast<const int32_t*>(raw.data_ptr()), dst, src, runs);
      else strided_rows_raw<float, int32_t, false>(spec, static_cast<const int32_t*>(raw.data_ptr()), dst, src, runs);
    } else {
      if (non_temporal) strided_rows_raw<float, int64_t, true>(spec, static_cast<const int64_t*>(raw.data_ptr()), dst, src, runs);
      else strided_rows_raw<float, int64_t, false>(spec, static_cast<const int64_t*>(raw.data_ptr()), dst, src, runs);
    }
  } else {
    const double* src = static_cast<const double*>(spec.transposed_self.data_ptr());
    double* dst = static_cast<double*>(out.data_ptr());
    if (idx32) {
      if (non_temporal) strided_rows_raw<double, int32_t, true>(spec, static_cast<const int32_t*>(raw.data_ptr()), dst, src, runs);
      else strided_rows_raw<double, int32_t, false>(spec, static_cast<const int32_t*>(raw.data_ptr()), dst, src, runs);
    } else {
      if (non_temporal) strided_rows_raw<double, int64_t, true>(spec, static_cast<const int64_t*>(raw.data_ptr()), dst, src, runs);
      else strided_rows_raw<double, int64_t, false>(spec, static_cast<const int64_t*>(raw.data_ptr()), dst, src, runs);
    }
  }
  return true;
}

void gather_fused(const IndexSpec& spec, at::Tensor& out) {  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(at::kHalf, at::kBFloat16, at::kBool,
      out.scalar_type(), "hpc_index_gather_fused", [&] {
    gather_fused<scalar_t>(spec, static_cast<scalar_t*>(out.data_ptr()),
        static_cast<const scalar_t*>(spec.transposed_self.data_ptr()));
  });
}

at::Tensor gather_mask(const at::Tensor& self, const at::Tensor& mask) {
  const int64_t md = mask.dim();
  const int64_t n = mask.numel();
  const uint8_t* mb = static_cast<const uint8_t*>(mask.data_ptr());
  int64_t A = 1;
  for (const auto d : c10::irange(md, self.dim())) A *= self.size(d);

  const int64_t grain = adaptive_grain(n);
  const int64_t nchunks = (n + grain - 1) / grain;
  std::vector<int64_t> counts(nchunks, 0);
  auto count_range = [&](int64_t lo, int64_t hi) {
    for (int64_t c = lo; c < hi; ++c) {
      const int64_t s = c * grain, e = std::min(n, s + grain);
      int64_t cnt = 0;
      for (int64_t p = s; p < e; ++p) cnt += (mb[p] != 0);
      counts[c] = cnt;
    }
  };
  // 小掩码并行计数被任务开销吞噬（200KB 实测并行反而慢 ~2x），串行扫描即可。
  if (n < (1LL << 20)) {
    count_range(0, nchunks);
  } else {
    at::parallel_for(0, nchunks, 1, count_range);
  }
  int64_t nnz = 0;
  std::vector<int64_t> offsets(nchunks + 1, 0);
  for (const auto c : c10::irange(nchunks)) {
    nnz += counts[c];
    offsets[c + 1] = nnz;
  }

  std::vector<int64_t> out_shape;
  out_shape.reserve(1 + self.dim() - md);
  out_shape.push_back(nnz);
  for (const auto d : c10::irange(md, self.dim())) out_shape.push_back(self.size(d));
  at::Tensor out = at::empty(out_shape, self.options());
  if (nnz == 0 || A == 0) return out;

  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(at::kHalf, at::kBFloat16, at::kBool,
      out.scalar_type(), "hpc_index_mask", [&] {
    const scalar_t* src = static_cast<const scalar_t*>(self.data_ptr());
    scalar_t* dst = static_cast<scalar_t*>(out.data_ptr());
    at::parallel_for(0, nchunks, 1, [&](int64_t lo, int64_t hi) {
      for (int64_t c = lo; c < hi; ++c) {
        int64_t w = offsets[c];
        const int64_t s = c * grain, e = std::min(n, s + grain);
        for (int64_t p = s; p < e; ++p) {
          if (mb[p] == 0) continue;
          const scalar_t* sp = src + p * A;
          scalar_t* dp = dst + w * A;
          if (A >= 4 && out.scalar_type() == at::kFloat) {
            copy_contig_sve<float, false>(
                reinterpret_cast<const float*>(sp), reinterpret_cast<float*>(dp), A);
            ++w;
            continue;
          }
          if (A >= 4 && out.scalar_type() == at::kDouble) {
            copy_contig_sve<double, false>(
                reinterpret_cast<const double*>(sp), reinterpret_cast<double*>(dp), A);
            ++w;
            continue;
          }
          for (const auto k : c10::irange(A)) dp[k] = sp[k];
          ++w;
        }
      }
    });
  });
  return out;
}

uint64_t strided_rows_count() { return g_strided_rows.load(std::memory_order_relaxed); }
uint64_t rowcopy_raw_count() { return g_rowcopy_raw.load(std::memory_order_relaxed); }
