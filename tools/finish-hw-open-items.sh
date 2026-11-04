#!/usr/bin/env bash
# 收尾两个未结论项：
#   A. 解析 HMAT / SLIT / SRAT，拿到固件声明的延迟/带宽等级；
#   B. 压封装内内存（node16-31），判定是共享链路还是各有通道。
#
# 用法: sudo bash tools/finish-hw-open-items.sh [输出文件]
# 输出默认写 hw-report-open-items-<时间>.txt（.gitignore 已排除），原始 ACPI 表与
# 探针源码放 hw-raw-open-items-<时间>/（同样被排除）。
#
# B 段会短时占用大量核（约 3 分钟，峰值 ~450 线程）；建议机器上没有别人跑大作业时做。

set -u

here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/.." && pwd)
ts=$(date +%Y%m%d-%H%M%S)
OUT=${1:-$repo/hw-report-open-items-$ts.txt}
RAW=$repo/hw-raw-open-items-$ts
mkdir -p "$RAW"
chmod 755 "$RAW"

exec > >(tee "$OUT") 2>&1

echo "# 硬件收尾采集  $(date -Is)"
echo "# 主机 $(hostname)  内核 $(uname -r)"
echo "# 输出 $OUT"

if [ "$(id -u)" != 0 ]; then
    echo "!! 需要 root：/sys/firmware/acpi/tables/* 是 0400 root"
    exit 1
fi
if ! command -v numactl >/dev/null || ! command -v gcc >/dev/null; then
    echo "!! 需要 numactl 与 gcc"
    exit 1
fi

echo
echo "==================== 0. 环境 ===================="
echo "loadavg: $(cut -d' ' -f1-3 /proc/loadavg)"
echo "nproc:   $(nproc)"
echo "cpuinfo_max_freq: $(cat /sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq 2>/dev/null || echo 未知) kHz"

echo
echo "==================== A. ACPI 表 ===================="
for t in HMAT SLIT SRAT; do
    if cp "/sys/firmware/acpi/tables/$t" "$RAW/$t" 2>/dev/null; then
        chmod 644 "$RAW/$t"
        echo "已复制 $t（$(stat -c%s "$RAW/$t") 字节）"
    else
        echo "!! 读不到 $t"
    fi
done
echo
python3 "$here/parse-acpi-hw.py" "$RAW" || echo "!! 解析失败，原始表在 $RAW"

echo
echo "==================== B1. 封装内内存的容量与占用 ===================="
have_pkg=0
for n in $(seq 16 31); do
    d=/sys/devices/system/node/node$n
    [ -d "$d" ] || continue
    have_pkg=1
    tot=$(awk '/^Node .*MemTotal/{print $4}' "$d/meminfo")
    free=$(awk '/^Node .*MemFree/{print $4}' "$d/meminfo")
    slab=$(awk '/^Node .*Slab:/{print $4}' "$d/meminfo")
    anon=$(awk '/^Node .*AnonPages/{print $4}' "$d/meminfo")
    printf 'node%-3s total=%5s MB free=%5s MB slab=%5s MB anon=%5s MB\n' \
        "$n" "$((tot / 1024))" "$((free / 1024))" "$((slab / 1024))" "$((anon / 1024))"
done
if [ "$have_pkg" = 0 ]; then
    echo "!! 本机没有 node16-31，B 段跳过"
    echo "# 完成（仅 A 段）"
    exit 0
fi

echo
echo "==================== B2. 编译探针 ===================="
BIN=$RAW/bin
mkdir -p "$BIN"
gcc -O2 -fopenmp -o "$BIN/memprobe" "$repo/benchmarks/memprobe.c" || { echo "!! memprobe 编译失败"; exit 1; }
cat > "$RAW/memwin.c" <<'EOF'
// 固定时间窗口的带宽探针：所有进程用同一个绝对时钟起点，在 [START, START+SEC)
// 内反复读自己的缓冲区，输出读到的字节数。聚合带宽 = Σ字节 / SEC。
// 与 docs/memory-hardware-primer.md 附录 A 同一份。
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
EOF
gcc -O2 -fopenmp -o "$BIN/memwin" "$RAW/memwin.c" || { echo "!! memwin 编译失败"; exit 1; }
echo "探针: $BIN/memprobe, $BIN/memwin"

cpus_of() { numactl -H | sed -n "s/^node $1 cpus: //p"; }
first_cpu() { cpus_of "$1" | awk '{print $1}'; }
last_cpu()  { cpus_of "$1" | awk '{print $NF}'; }

first0=$(first_cpu 0); last0=$(last_cpu 7)
first1=$(first_cpu 8); last1=$(last_cpu 15)
slice0=$(( (last0 - first0 + 1) / 8 )); [ "$slice0" -gt 32 ] && slice0=32
slice1=$(( (last1 - first1 + 1) / 8 )); [ "$slice1" -gt 32 ] && slice1=32
echo "socket0 核 ${first0}-${last0}（每进程 ${slice0} 核），socket1 核 ${first1}-${last1}（每进程 ${slice1} 核）"

lat() { # cpus node -> 64MB 依赖链延迟 ns
    numactl --physcpubind="$1" --membind="$2" "$BIN/memprobe" chase 65536 2000000 \
        | sed -n 's/.*ns\/access=\([0-9.]*\).*/\1/p'
}

mwin() { # cpus node sec threads -> 两次取最好 GB/s
    local cpus=$1 node=$2 sec=$3 nt=$4 best=0 out start
    for _ in 1 2; do
        start=$(awk -v t="$(date +%s.%N)" 'BEGIN{printf "%.6f", t+2}')
        out=$(numactl --physcpubind="$cpus" --membind="$node" "$BIN/memwin" 128 "$sec" "$start" "$nt")
        best=$(awk -v a="$best" -v b="$out" -v s="$sec" 'BEGIN{v=b/s/1e9; print (v>a?v:a)}')
    done
    printf '%.1f' "$best"
}

mwin_multi() { # sec threads spec... (spec=cpus:node) -> 聚合 GB/s
    local sec=$1 nt=$2; shift 2
    local tmp start i=0 pids=()
    tmp=$(mktemp -d)
    start=$(awk -v t="$(date +%s.%N)" 'BEGIN{printf "%.6f", t+3}')
    for spec in "$@"; do
        numactl --physcpubind="${spec%%:*}" --membind="${spec##*:}" \
            "$BIN/memwin" 128 "$sec" "$start" "$nt" > "$tmp/$i.out" 2>/dev/null &
        pids+=($!)
        i=$((i + 1))
    done
    for p in "${pids[@]}"; do wait "$p"; done
    awk -v s="$sec" '{b += $1} END{printf "%.1f", b/s/1e9}' "$tmp"/*.out
    rm -rf "$tmp"
}

usable() { [ "$(awk '/^Node .*MemFree/{print $4}' "/sys/devices/system/node/node$1/meminfo")" -gt 524288 ]; }

echo
echo "==================== B3. 单 node：延迟与饱和带宽 ===================="
echo "（延迟用 64MB 足迹，避开 L2 混合；带宽 32 线程、128MB 缓冲区、2s 窗口取两次最好）"
printf '%-7s %-6s %-10s %-11s %s\n' node 归属socket 延迟ns 带宽GB/s "距离 d(5)/d(13)"
pkg0=(); pkg1=()
for n in $(seq 16 31); do
    d5=$(cut -d' ' -f$((n + 1)) "/sys/devices/system/node/node5/distance")
    d13=$(cut -d' ' -f$((n + 1)) "/sys/devices/system/node/node13/distance")
    if [ "${d5:-999}" -le "${d13:-999}" ]; then side=0; base=$first0; sl=$slice0; else side=1; base=$first1; sl=$slice1; fi
    if usable "$n"; then
        cpus="$base-$((base + sl - 1))"
        l=$(lat "$base" "$n")
        b=$(mwin "$cpus" "$n" 2 "$sl")
        printf 'node%-3s %-9s %-10s %-11s %s/%s\n' "$n" "$side" "$l" "$b" "$d5" "$d13"
        if [ "$side" = 0 ]; then pkg0+=("$n"); else pkg1+=("$n"); fi
    else
        printf 'node%-3s %-9s %-10s %-11s %s/%s  空闲<512MB，跳过\n' "$n" "$side" "-" "-" "$d5" "$d13"
    fi
done
echo "socket0 侧可用: ${pkg0[*]:-无}"
echo "socket1 侧可用: ${pkg1[*]:-无}"
echo "本地参照（同一批核，内存放自己的 node）：node0=$(lat "$first0" 0)ns / $(mwin "$first0-$((first0 + slice0 - 1))" 0 2 "$slice0")GBps，node8=$(lat "$first1" 8)ns / $(mwin "$first1-$((first1 + slice1 - 1))" 8 2 "$slice1")GBps"

echo
echo "==================== B4. 核 → 内存 延迟矩阵 ===================="
echo "（行=发起核所在 node，列=内存所在 node；单位 ns，64MB 足迹）"
cols="0 4 8 12 5 13 17 18 19 20 21 22 23 24 25 26 27 28 29 30"
printf '%-6s' 发起
for c in $cols; do printf '%6s' "$c"; done
echo
for a in 0 4 8 12 5 13; do
    printf '%-6s' "$a"
    for m in $cols; do
        v=$(lat "$(first_cpu "$a")" "$m")
        printf '%6s' "${v%%.*}"
    done
    echo
done

echo
echo "==================== B5. 单 node 的线程扫描 ===================="
sweep() { # base node
    local base=$1 n=$2 t cpus
    for t in 8 16 32 64; do
        cpus="$base-$((base + t - 1))"
        echo "node$n ${t}t: $(mwin "$cpus" "$n" 2 "$t") GB/s"
    done
}
[ "${#pkg0[@]}" -gt 0 ] && sweep "$first0" "${pkg0[0]}"
[ "${#pkg1[@]}" -gt 0 ] && sweep "$first1" "${pkg1[0]}"

echo
echo "==================== B6. 多 node 扩展（每 node 一个进程，各 32 线程） ===================="
spread() { # label base slice nodes...
    local label=$1 base=$2 sl=$3 nt=$4; shift 4
    local specs=() i=0 lo
    for n in "$@"; do
        lo=$((base + i * sl))
        specs+=("$lo-$((lo + sl - 1)):$n")
        i=$((i + 1))
    done
    echo "$label（nodes=$*）：$(mwin_multi 2 "$nt" "${specs[@]}") GB/s"
}
if [ "${#pkg0[@]}" -gt 0 ]; then
    for k in 1 2 4; do
        [ "${#pkg0[@]}" -ge "$k" ] && spread "socket0 ${k} node" "$first0" "$slice0" 32 "${pkg0[@]:0:$k}"
    done
    spread "socket0 ${#pkg0[@]} node" "$first0" "$slice0" 32 "${pkg0[@]}"
    same=()
    for _ in "${pkg0[@]}"; do same+=("$first0-$((first0 + slice0 - 1)):${pkg0[0]}"); done
    echo "对照：${#pkg0[@]} 个进程全压在 node${pkg0[0]} 上：$(mwin_multi 2 32 "${same[@]}") GB/s"
fi
if [ "${#pkg1[@]}" -gt 0 ]; then
    spread "socket1 ${#pkg1[@]} node" "$first1" "$slice1" 32 "${pkg1[@]}"
    same=()
    for _ in "${pkg1[@]}"; do same+=("$first1-$((first1 + slice1 - 1)):${pkg1[0]}"); done
    echo "对照：${#pkg1[@]} 个进程全压在 node${pkg1[0]} 上：$(mwin_multi 2 32 "${same[@]}") GB/s"
fi

echo
echo "==================== B7. 两个 socket 同时压 ===================="
specs=()
i=0
for n in "${pkg0[@]}"; do lo=$((first0 + i * slice0)); specs+=("$lo-$((lo + slice0 - 1)):$n"); i=$((i + 1)); done
i=0
for n in "${pkg1[@]}"; do lo=$((first1 + i * slice1)); specs+=("$lo-$((lo + slice1 - 1)):$n"); i=$((i + 1)); done
[ "${#specs[@]}" -gt 0 ] && echo "共 ${#specs[@]} 个进程 × 32t：$(mwin_multi 2 32 "${specs[@]}") GB/s"

echo
echo "==================== 完成 ===================="
echo "raw: $RAW"
echo "txt: $OUT"
