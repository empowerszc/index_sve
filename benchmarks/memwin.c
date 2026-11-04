// 固定时间窗口的带宽探针：所有进程用同一个绝对时钟起点，在 [START, START+SEC)
// 内反复读自己的缓冲区，输出读到的字节数。聚合带宽 = Σ字节 / SEC。
// 与 docs/memory-hardware-primer.md 附录 A 同一份；tools/finish-hw-open-items.sh
// 与 tools/probe-onpkg-bw.py 都用它。
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <omp.h>

static double re_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return ts.tv_sec + 1e-9 * ts.tv_nsec;
}
static double mono_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s BUF_MB SEC START_EPOCH [THREADS]\n", argv[0]);
        return 1;
    }
    size_t size = (size_t)atol(argv[1]) * 1024ULL * 1024ULL;
    double sec = atof(argv[2]), start = atof(argv[3]);
    int nt = argc > 4 ? atoi(argv[4]) : 1;
    size_t n = size / 8;
    uint64_t *a = aligned_alloc(64, size);
    if (!a) { fprintf(stderr, "alloc failed\n"); return 1; }

    #pragma omp parallel num_threads(nt)
    {
        size_t lo = (size_t)((double)n * omp_get_thread_num() / nt);
        size_t hi = (size_t)((double)n * (omp_get_thread_num() + 1) / nt);
        for (size_t i = lo; i < hi; i++) a[i] = i * 3 + 1;
    }

    while (re_now() < start) { }
    double t0 = mono_now();
    unsigned long long bytes = 0;
    double deadline = sec;
    while (mono_now() - t0 < deadline) {
        #pragma omp parallel num_threads(nt) reduction(+:bytes)
        {
            size_t lo = (size_t)((double)n * omp_get_thread_num() / nt);
            size_t hi = (size_t)((double)n * (omp_get_thread_num() + 1) / nt);
            uint64_t s0 = 0, s1 = 0, s2 = 0, s3 = 0;
            for (size_t i = lo; i + 3 < hi; i += 4) {
                s0 += a[i]; s1 += a[i + 1]; s2 += a[i + 2]; s3 += a[i + 3];
            }
            bytes += (hi - lo) * 8ULL;
            if (omp_get_thread_num() == 0) a[0] += s0 + s1 + s2 + s3;
        }
    }
    printf("%.0f\n", (double)bytes);
    return 0;
}
