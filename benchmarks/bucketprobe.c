// bucketprobe：把桶化输出段的两种写法隔离出来对比。
//
// 目的：回答"离探针下界 7-10 倍的那段，是编译器/写法问题（可以改），还是内存系统
// 的硬账（改不动）"。两种写法除了 pass 2 的循环体以外完全相同（同一份 pass 1、
// 同样的串行建表、同样的 chunk 划分），所以差值只能归因到"批量化"。
//
//   cur    : 逐元素  dst[k] = table[idx[k]-mn]        （当前内核的写法）
//   b16/b32: 先算好 16/32 个下标，再一起发 16/32 条独立 load（H2b 的写法）
//
// 用法：./bucketprobe N DOM THREADS REPS [cur|b16|b32|all]
//   N = 元素数（K），DOM = 值域大小（表长），THREADS = 线程数
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <limits.h>
#include <omp.h>
#include <sys/mman.h>

static double now(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static uint64_t xs(uint64_t *s) {
  uint64_t x = *s;
  x ^= x << 13;
  x ^= x >> 7;
  x ^= x << 17;
  return *s = x;
}

static int32_t *g_idx;
static float *g_src, *g_dst, *g_table;
static int64_t g_n;
static int g_dom;
static int64_t g_chunk;

// pass 1：与原内核同构（读下标、负索引归一、求 min/max）
static void pass1(int64_t *mn, int64_t *mx) {
  int64_t lmn = INT64_MAX, lmx = INT64_MIN;
#pragma omp parallel num_threads(omp_get_max_threads())
  {
#pragma omp for schedule(static)
    for (int64_t c = 0; c < g_n; c += g_chunk) {
      int64_t hi = c + g_chunk < g_n ? c + g_chunk : g_n;
      int64_t a = INT64_MAX, b = INT64_MIN;
      for (int64_t k = c; k < hi; ++k) {
        int64_t v = g_idx[k];
        a = v < a ? v : a;
        b = v > b ? v : b;
      }
      if (a < lmn) lmn = a;
      if (b > lmx) lmx = b;
    }
  }
  *mn = lmn;
  *mx = lmx;
}

static void run_cur(void) {
  int64_t mn, mx;
  pass1(&mn, &mx);
  int64_t dom = mx - mn + 1;
  for (int64_t v = 0; v < dom; ++v) g_table[v] = g_src[v];
  const int64_t n = g_n, chunk = g_chunk;
#pragma omp parallel for schedule(static)
  for (int64_t c = 0; c < n; c += chunk) {
    int64_t hi = c + chunk < n ? c + chunk : n;
    for (int64_t k = c; k < hi; ++k) g_dst[k] = g_table[g_idx[k] - mn];
  }
}

// cur_real：完全照抄 bucket_gather_pure_raw 的循环体（pass 1 带越界检查 + 负索引
// 归一，pass 2 再归一一次）。与 cur 的唯一差别就是这两行——用来判断"内核比探针慢
// 2 倍"是不是检查本身的账。
static volatile int g_bad;
static void run_cur_real(void) {
  int64_t mn, mx;
  const int64_t size = g_dom;
  int64_t lmn = INT64_MAX, lmx = INT64_MIN;
#pragma omp parallel num_threads(omp_get_max_threads())
  {
#pragma omp for schedule(static)
    for (int64_t c = 0; c < g_n; c += g_chunk) {
      int64_t hi = c + g_chunk < g_n ? c + g_chunk : g_n;
      int64_t a = INT64_MAX, b = INT64_MIN;
      for (int64_t k = c; k < hi; ++k) {
        int64_t v = g_idx[k];
        if (v < -size || v >= size) { g_bad = 1; v = 0; }   // TORCH_CHECK_INDEX
        if (v < 0) v += size;
        a = v < a ? v : a;
        b = v > b ? v : b;
      }
      if (a < lmn) lmn = a;
      if (b > lmx) lmx = b;
    }
  }
  mn = lmn; mx = lmx;
  int64_t dom = mx - mn + 1;
  for (int64_t v = 0; v < dom; ++v) g_table[v] = g_src[v];
  const int64_t n = g_n, chunk = g_chunk;
#pragma omp parallel for schedule(static)
  for (int64_t c = 0; c < n; c += chunk) {
    int64_t hi = c + chunk < n ? c + chunk : n;
    for (int64_t k = c; k < hi; ++k) {
      int64_t v = g_idx[k];
      if (v < 0) v += size;
      g_dst[k] = g_table[v - mn];
    }
  }
}

// real_chk / real_sign：把 cur_real 的两项开销拆开（越界检查 / 负索引归一），
// 用来判断补丁该动哪一个。
#define REAL_VARIANT(NAME, DOCHK, DOSIGN)                                     \
  static void NAME(void) {                                                    \
    const int64_t size = g_dom;                                               \
    int64_t lmn = INT64_MAX, lmx = INT64_MIN;                                 \
    _Pragma("omp parallel num_threads(omp_get_max_threads())")                \
    {                                                                         \
      _Pragma("omp for schedule(static)")                                     \
      for (int64_t c = 0; c < g_n; c += g_chunk) {                            \
        int64_t hi = c + g_chunk < g_n ? c + g_chunk : g_n;                   \
        int64_t a = INT64_MAX, b = INT64_MIN;                                 \
        for (int64_t k = c; k < hi; ++k) {                                    \
          int64_t v = g_idx[k];                                               \
          if (DOCHK && (uint64_t)(v + size) >= (uint64_t)(2 * size)) {         \
            g_bad = 1;                                                        \
            v = 0;                                                            \
          }                                                                   \
          if (DOSIGN && v < 0) v += size;                                     \
          a = v < a ? v : a;                                                  \
          b = v > b ? v : b;                                                  \
        }                                                                     \
        if (a < lmn) lmn = a;                                                 \
        if (b > lmx) lmx = b;                                                 \
      }                                                                       \
    }                                                                         \
    int64_t mn = lmn;                                                         \
    int64_t dom = lmx - lmn + 1;                                              \
    for (int64_t v = 0; v < dom; ++v) g_table[v] = g_src[v];                  \
    const int64_t n = g_n, chunk = g_chunk;                                   \
    _Pragma("omp parallel for schedule(static)")                              \
    for (int64_t c = 0; c < n; c += chunk) {                                  \
      int64_t hi = c + chunk < n ? c + chunk : n;                             \
      for (int64_t k = c; k < hi; ++k) {                                      \
        int64_t v = g_idx[k];                                                 \
        if (DOSIGN && v < 0) v += size;                                       \
        g_dst[k] = g_table[v - mn];                                           \
      }                                                                       \
    }                                                                         \
  }
REAL_VARIANT(run_chk, 1, 0)
REAL_VARIANT(run_sign, 0, 1)

// bl_sign：负索引归一写成 csel（无分支）；half_sign：只在 pass 1 归一（pass 2 靠
// "mn>=0 则全非负"这个事实跳过）。这两个就是候选补丁的两种写法。
#define SIGN_VARIANT(NAME, P1, P2, BRANCHLESS)                                \
  static void NAME(void) {                                                    \
    const int64_t size = g_dom;                                               \
    int64_t lmn = INT64_MAX, lmx = INT64_MIN;                                 \
    _Pragma("omp parallel num_threads(omp_get_max_threads())")                \
    {                                                                         \
      _Pragma("omp for schedule(static)")                                     \
      for (int64_t c = 0; c < g_n; c += g_chunk) {                            \
        int64_t hi = c + g_chunk < g_n ? c + g_chunk : g_n;                   \
        int64_t a = INT64_MAX, b = INT64_MIN;                                 \
        for (int64_t k = c; k < hi; ++k) {                                    \
          int64_t v = g_idx[k];                                               \
          if (P1) v += BRANCHLESS ? (v < 0 ? size : 0) : (v < 0 ? size : 0);  \
          a = v < a ? v : a;                                                  \
          b = v > b ? v : b;                                                  \
        }                                                                     \
        if (a < lmn) lmn = a;                                                 \
        if (b > lmx) lmx = b;                                                 \
      }                                                                       \
    }                                                                         \
    int64_t mn = lmn;                                                         \
    int64_t dom = lmx - lmn + 1;                                              \
    for (int64_t v = 0; v < dom; ++v) g_table[v] = g_src[v];                  \
    const int64_t n = g_n, chunk = g_chunk;                                   \
    _Pragma("omp parallel for schedule(static)")                              \
    for (int64_t c = 0; c < n; c += chunk) {                                  \
      int64_t hi = c + chunk < n ? c + chunk : n;                             \
      for (int64_t k = c; k < hi; ++k) {                                      \
        int64_t v = g_idx[k];                                                 \
        if (P2) v += (v < 0 ? size : 0);                                      \
        g_dst[k] = g_table[v - mn];                                           \
      }                                                                       \
    }                                                                         \
  }
SIGN_VARIANT(run_half, 1, 0, 0)
SIGN_VARIANT(run_bl, 1, 1, 1)

// nop1：候选补丁的上界——源本身 <= MAX_DOMAIN 时，表可以直接按 [0, numel) 建，
// 于是**整趟 pass 1（扫描 min/max + 一次 fork-join）都不需要**，越界检查与负索引
// 归一折进 pass 2 的一条比较 + 一条 csel。这一步改的是算法（少一趟），不是写法。
static void run_nop1(void) {
  const int64_t size = g_dom;
  for (int64_t v = 0; v < size; ++v) g_table[v] = g_src[v];
  const int64_t n = g_n, chunk = g_chunk;
#pragma omp parallel for schedule(static)
  for (int64_t c = 0; c < n; c += chunk) {
    int64_t hi = c + chunk < n ? c + chunk : n;
    for (int64_t k = c; k < hi; ++k) {
      int64_t v = g_idx[k];
      if ((uint64_t)(v + size) >= (uint64_t)(2 * size)) { g_bad = 1; v = 0; }
      v += (v < 0 ? size : 0);
      g_dst[k] = g_table[v];
    }
  }
}

// cur_slow：把内核里那些"看起来免费"的每调用开销补回来——表用 malloc 新建
// （等价 std::vector<T>(dom) 的零初始化 + free）、输出每次 mmap 新页（等价新张量
// 的 first-touch）。用来判断"探针与内核差 2 倍"到底在不在循环体里。
static void run_cur_slow(void) {
  int64_t mn, mx;
  pass1(&mn, &mx);
  int64_t dom = mx - mn + 1;
  float *table = calloc((size_t)dom, sizeof(float));
  for (int64_t v = 0; v < dom; ++v) table[v] = g_src[v];
  float *dst = mmap(NULL, (size_t)g_n * sizeof(float), PROT_READ | PROT_WRITE,
                    MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  const int64_t n = g_n, chunk = g_chunk;
#pragma omp parallel for schedule(static)
  for (int64_t c = 0; c < n; c += chunk) {
    int64_t hi = c + chunk < n ? c + chunk : n;
    for (int64_t k = c; k < hi; ++k) dst[k] = table[g_idx[k] - mn];
  }
  g_dst[g_n / 2] = dst[g_n / 2];
  munmap(dst, (size_t)g_n * sizeof(float));
  free(table);
}

// b16/b32：先取一批下标，再一起查表；两批之间的 load 互相独立
#define BATCH(NAME, W)                                                        \
  static void NAME(void) {                                                    \
    int64_t mn, mx;                                                           \
    pass1(&mn, &mx);                                                          \
    int64_t dom = mx - mn + 1;                                                \
    for (int64_t v = 0; v < dom; ++v) g_table[v] = g_src[v];                   \
    const int64_t n = g_n, chunk = g_chunk;                                   \
    _Pragma("omp parallel for schedule(static)")                              \
    for (int64_t c = 0; c < n; c += chunk) {                                  \
      int64_t hi = c + chunk < n ? c + chunk : n;                             \
      int64_t k = c;                                                          \
      for (; k + W <= hi; k += W) {                                           \
        float r[W];                                                           \
        for (int j = 0; j < W; ++j) r[j] = g_table[g_idx[k + j] - mn];        \
        for (int j = 0; j < W; ++j) g_dst[k + j] = r[j];                      \
      }                                                                       \
      for (; k < hi; ++k) g_dst[k] = g_table[g_idx[k] - mn];                  \
    }                                                                         \
  }

BATCH(run_b16, 16)
BATCH(run_b32, 32)

int main(int argc, char **argv) {
  if (argc < 5) {
    fprintf(stderr, "usage: %s N DOM THREADS REPS [cur|b16|b32|all]\n", argv[0]);
    return 1;
  }
  g_n = atoll(argv[1]);
  g_dom = atoi(argv[2]);
  int t = atoi(argv[3]);
  int reps = atoi(argv[4]);
  const char *which = argc > 5 ? argv[5] : "all";
  omp_set_num_threads(t);
  g_chunk = g_n / (t * 8);
  if (g_chunk < 64) g_chunk = 64;

  g_idx = aligned_alloc(64, g_n * sizeof(int32_t));
  g_src = aligned_alloc(64, g_dom * sizeof(float));
  g_table = aligned_alloc(64, g_dom * sizeof(float));
  g_dst = aligned_alloc(64, g_n * sizeof(float));
  uint64_t s = 88172645463325252ULL;
  for (int64_t k = 0; k < g_n; ++k) g_idx[k] = (int32_t)(xs(&s) % (uint64_t)g_dom);
  for (int i = 0; i < g_dom; ++i) g_src[i] = (float)(i + 1);

  struct { const char *name; void (*fn)(void); } cases[] = {
      {"cur", run_cur}, {"b16", run_b16}, {"b32", run_b32},
      {"cur_real", run_cur_real}, {"chk_only", run_chk}, {"sign_only", run_sign},
      {"half_sign", run_half}, {"bl_sign", run_bl}, {"nop1", run_nop1}};
  for (int i = 0; i < 9; ++i) {
    if (strcmp(which, "all") && strcmp(which, cases[i].name)) continue;
    double best = 1e18;
    for (int r = 0; r < reps; ++r) {
      double t0 = now();
      cases[i].fn();
      double dt = now() - t0;
      if (dt < best) best = dt;
    }
    printf("%-4s n=%lld dom=%d t=%d  total=%.2fus  per_el=%.2fns  sink=%.0f\n",
           cases[i].name, (long long)g_n, g_dom, t, best * 1e6,
           best * 1e9 / (double)g_n, (double)g_dst[g_n / 2]);
  }
  free(g_idx); free(g_src); free(g_table); free(g_dst);
  return 0;
}
