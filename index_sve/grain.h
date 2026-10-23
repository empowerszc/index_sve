#pragma once
#include <ATen/Parallel.h>
#include <algorithm>
#include <cstdint>

// 自适应并行粒度：按元素数与线程数缩放 chunk 大小，保证小 K 也能占满所有线程，
// 大 K 不因过大 grain 而退化。grain = max(64, min(n, 65536, n / (nt * 8)))；
// 下限 64 防止高线程把极小任务切成单元素 chunk（fork-join 任务风暴）。
inline int64_t adaptive_grain(int64_t n) {
  constexpr int64_t MAX_GRAIN = 65536;
  constexpr int64_t MIN_GRAIN = 64;
  const int64_t nt = at::get_num_threads();
  if (nt <= 1) return std::max<int64_t>(1, n);
  int64_t grain = std::max<int64_t>(MIN_GRAIN, n / (nt * 8));
  return std::min<int64_t>(grain, MAX_GRAIN);
}
