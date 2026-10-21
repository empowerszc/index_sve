#include "linearize.h"
#include "grain.h"
#include <ATen/Parallel.h>
#include <c10/util/Exception.h>
#include <c10/util/irange.h>

namespace {

void linearize_i64(const IndexSpec& spec, int64_t* idx) {
  const int64_t K = spec.K;
  const int64_t nidx = spec.indexed_dims.size();
  const std::vector<int64_t>& bshape = spec.broadcast_shape;
  const int64_t nb = bshape.size();
  std::vector<int64_t> bfactor(nb, 1);          // 行优先混合进制因子（A7）
  for (int64_t i = nb - 2; i >= 0; --i) {
    bfactor[i] = bfactor[i + 1] * bshape[i + 1];
  }
  // 预提取各下标张量的指针/stride/esize/长整型标志，避免内层循环每次访问
  // at::Tensor 方法（引用计数 + 虚调用），这是 linearize 单线程开销的根因。
  std::vector<const void*> ind_ptr(nidx);
  std::vector<const int64_t*> ind_stride(nidx);
  std::vector<int64_t> ind_esize(nidx);
  std::vector<bool> ind_long(nidx);
  bool all_full_contig = (nb == 1);
  for (const auto j : c10::irange(nidx)) {
    const at::Tensor& ind = spec.indices[j];
    ind_ptr[j] = ind.data_ptr();
    ind_stride[j] = ind.strides().data();
    ind_esize[j] = ind.element_size();
    ind_long[j] = (ind.dtype() == at::kLong);
    if (ind.numel() != K || !ind.is_contiguous()) all_full_contig = false;
  }
  // nb==1 快路径：所有索引全长连续（无广播展开），解码退化为直读 idx_j[k]。
  if (all_full_contig) {
    at::parallel_for(0, K, adaptive_grain(K), [&](int64_t lo, int64_t hi) {
      for (int64_t k = lo; k < hi; ++k) {
        int64_t off = 0;
        for (const auto j : c10::irange(nidx)) {
          const char* p = static_cast<const char*>(ind_ptr[j]) + k * ind_esize[j];
          int64_t v = ind_long[j]
              ? *reinterpret_cast<const int64_t*>(p)
              : *reinterpret_cast<const int32_t*>(p);
          const int64_t size = spec.indexed_sizes[j];
          TORCH_CHECK_INDEX(v >= -size && v < size,
              "index ", v, " is out of bounds for dimension ", j, " with size ", size);
          if (v < 0) v += size;
          off += v * spec.indexed_strides[j];
        }
        idx[k] = off;
      }
    });
    return;
  }
  at::parallel_for(0, K, adaptive_grain(K), [&](int64_t lo, int64_t hi) {
    for (int64_t k = lo; k < hi; ++k) {
      int64_t r = k;
      int64_t off = 0;
      for (const auto j : c10::irange(nidx)) {
        const int64_t esize = ind_esize[j];
        const int64_t* ind_st = ind_stride[j];
        int64_t eoff = 0;
        int64_t rr = r;
        for (const auto d : c10::irange(nb)) {
          int64_t c = (bshape[d] > 0) ? (rr / bfactor[d]) % bshape[d] : 0;
          eoff += c * ind_st[d];
          if (bshape[d] > 0) rr %= bfactor[d];
        }
        const char* p = static_cast<const char*>(ind_ptr[j]) + eoff * esize;
        int64_t v = ind_long[j]
            ? *reinterpret_cast<const int64_t*>(p)
            : *reinterpret_cast<const int32_t*>(p);
        const int64_t size = spec.indexed_sizes[j];
        TORCH_CHECK_INDEX(v >= -size && v < size,
            "index ", v, " is out of bounds for dimension ", j, " with size ", size);
        if (v < 0) v += size;
        off += v * spec.indexed_strides[j];
      }
      idx[k] = off;
    }
  });
}

}  // namespace

void linearize(const IndexSpec& spec, at::Tensor& idx_out) {
  if (idx_out.scalar_type() == at::kLong) {
    linearize_i64(spec, static_cast<int64_t*>(idx_out.data_ptr()));
  } else {
    TORCH_INTERNAL_ASSERT(idx_out.scalar_type() == at::kInt);
    // 直接写 int32，不再经 K×8B int64 tmp（RESULTS §6 的遗留改进）：use_i32 保证
    // 偏移 < 2^31，值域安全。
    const int64_t K = spec.K;
    const int64_t nidx = spec.indexed_dims.size();
    const std::vector<int64_t>& bshape = spec.broadcast_shape;
    const int64_t nb = bshape.size();
    std::vector<int64_t> bfactor(nb, 1);          // 行优先混合进制因子（A7）
    for (int64_t i = nb - 2; i >= 0; --i) {
      bfactor[i] = bfactor[i + 1] * bshape[i + 1];
    }
    std::vector<const void*> ind_ptr(nidx);
    std::vector<const int64_t*> ind_stride(nidx);
    std::vector<int64_t> ind_esize(nidx);
    std::vector<bool> ind_long(nidx);
    bool all_full_contig = (nb == 1);
    for (const auto j : c10::irange(nidx)) {
      const at::Tensor& ind = spec.indices[j];
      ind_ptr[j] = ind.data_ptr();
      ind_stride[j] = ind.strides().data();
      ind_esize[j] = ind.element_size();
      ind_long[j] = (ind.dtype() == at::kLong);
      if (ind.numel() != K || !ind.is_contiguous()) all_full_contig = false;
    }
    int32_t* dst = static_cast<int32_t*>(idx_out.data_ptr());
    if (all_full_contig) {
      at::parallel_for(0, K, adaptive_grain(K), [&](int64_t lo, int64_t hi) {
        for (int64_t k = lo; k < hi; ++k) {
          int64_t off = 0;
          for (const auto j : c10::irange(nidx)) {
            const char* p = static_cast<const char*>(ind_ptr[j]) + k * ind_esize[j];
            int64_t v = ind_long[j]
                ? *reinterpret_cast<const int64_t*>(p)
                : *reinterpret_cast<const int32_t*>(p);
            const int64_t size = spec.indexed_sizes[j];
            TORCH_CHECK_INDEX(v >= -size && v < size,
                "index ", v, " is out of bounds for dimension ", j, " with size ", size);
            if (v < 0) v += size;
            off += v * spec.indexed_strides[j];
          }
          dst[k] = static_cast<int32_t>(off);
        }
      });
      return;
    }
    at::parallel_for(0, K, adaptive_grain(K), [&](int64_t lo, int64_t hi) {
      for (int64_t k = lo; k < hi; ++k) {
        int64_t r = k;
        int64_t off = 0;
        for (const auto j : c10::irange(nidx)) {
          const int64_t esize = ind_esize[j];
          const int64_t* ind_st = ind_stride[j];
          int64_t eoff = 0;
          int64_t rr = r;
          for (const auto d : c10::irange(nb)) {
            int64_t c = (bshape[d] > 0) ? (rr / bfactor[d]) % bshape[d] : 0;
            eoff += c * ind_st[d];
            if (bshape[d] > 0) rr %= bfactor[d];
          }
          const char* p = static_cast<const char*>(ind_ptr[j]) + eoff * esize;
          int64_t v = ind_long[j]
              ? *reinterpret_cast<const int64_t*>(p)
              : *reinterpret_cast<const int32_t*>(p);
          const int64_t size = spec.indexed_sizes[j];
          TORCH_CHECK_INDEX(v >= -size && v < size,
              "index ", v, " is out of bounds for dimension ", j, " with size ", size);
          if (v < 0) v += size;
          off += v * spec.indexed_strides[j];
        }
        dst[k] = static_cast<int32_t>(off);
      }
    });
  }
}
