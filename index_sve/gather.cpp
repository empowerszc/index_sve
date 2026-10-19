#include "gather.h"
#include "grain.h"
#include <ATen/Parallel.h>
#include <algorithm>
#include <arm_sve.h>
#include <c10/util/irange.h>
#include <type_traits>

namespace {

constexpr int64_t PREFETCH_DIST = 16;  // 微基准扫描 {0,16,32,64} 后定的默认

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
  at::parallel_for(0, B * K, adaptive_grain(B * K), [&](int64_t lo, int64_t hi) {
    for (int64_t p = lo; p < hi; ++p) {
      const int64_t r = p / K, k = p % K;
      int64_t off = spec.row_off[r];
      for (const auto j : c10::irange(nidx)) {
        const at::Tensor& ind = spec.indices[j];
        const int64_t esize = ind.element_size();
        int64_t eoff = 0;
        int64_t rr = k;
        for (const auto d : c10::irange(nb)) {   // A7：按 broadcast_shape 坐标读元素
          int64_t c = (bshape[d] > 0) ? (rr / bfactor[d]) % bshape[d] : 0;
          eoff += c * ind.stride(d);
          if (bshape[d] > 0) rr %= bfactor[d];
        }
        const char* ptr = static_cast<const char*>(ind.data_ptr()) + eoff * esize;
        int64_t v = (ind.dtype() == at::kLong)
            ? *reinterpret_cast<const int64_t*>(ptr)
            : *reinterpret_cast<const int32_t*>(ptr);
        const int64_t size = spec.indexed_sizes[j];
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

// M1 单索引大块 skip-linearize：不物化 idx，直接在 gather 阶段读原始下标、内联边界
// 检查与负索引规整（等价 linearize 内循环），随后对 A 连续块做 SVE rowcopy。只有
// spec.after_contiguous 才启用。out 行序 = 原下标序（只改变 src 读序由 T2 阶段 2 处理）。
template <typename T, typename IdxT, bool NT>
void rowcopy_raw_single(const IndexSpec& spec, const IdxT* raw, T* dst, const T* src) {
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
      copy_contig_sve<T, NT>(s, d, A);
    }
  });
}

void gather_rowcopy_raw(const IndexSpec& spec, at::Tensor& out, bool non_temporal) {
  const at::Tensor& raw = spec.indices[0];
  const bool idx32 = raw.scalar_type() == at::kInt;
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(at::kHalf, at::kBFloat16, at::kBool,
      out.scalar_type(), "hpc_index_rowcopy_raw", [&] {
    const scalar_t* src = static_cast<const scalar_t*>(spec.transposed_self.data_ptr());
    scalar_t* dst = static_cast<scalar_t*>(out.data_ptr());
    if (idx32) {
      rowcopy_raw_single<scalar_t, int32_t, false>(
          spec, static_cast<const int32_t*>(raw.data_ptr()), dst, src);
    } else {
      rowcopy_raw_single<scalar_t, int64_t, false>(
          spec, static_cast<const int64_t*>(raw.data_ptr()), dst, src);
    }
  });
}
