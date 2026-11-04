#!/usr/bin/env bash
# 采集本机 CPU / 内存 / NUMA 硬件信息，汇总成一个文本文件。
#
# 用法：
#   sudo bash tools/collect-hw-info.sh              # 输出到当前目录
#   sudo bash tools/collect-hw-info.sh /tmp/opencode # 输出到指定目录
#   sudo bash tools/collect-hw-info.sh --redact      # 去掉序列号/UUID 等
#
# 只读，不改任何系统设置。跑完把输出的 .txt 路径给我即可。

set -u

REDACT=0
OUT_DIR="$PWD"
for a in "$@"; do
  case "$a" in
    --redact) REDACT=1 ;;
    -*) echo "未知参数: $a" >&2; exit 2 ;;
    *) OUT_DIR="$a" ;;
  esac
done

if [ "$(id -u)" -ne 0 ]; then
  echo "需要 root 才能读 DMI / ACPI 表。请这样跑：" >&2
  echo "  sudo bash $0 $*" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"
TS=$(date +%Y%m%d-%H%M%S)
OUT="$OUT_DIR/hw-report-$TS.txt"
RAW="$OUT_DIR/hw-raw-$TS"
mkdir -p "$RAW"

exec 3>&1            # 保存终端 stdout
exec > "$OUT" 2>&1   # 之后所有输出进报告

sec() { printf '\n\n========== %s ==========\n' "$1"; }

# 跑一条命令；命令不存在时说明一下，不让整个脚本挂掉。
run() {
  local desc="$1" cmd="$2"
  printf '\n----- %s -----\n' "$desc"
  printf '$ %s\n' "$cmd"
  if ! timeout 90 bash -c "$cmd"; then
    echo "[命令退出码 $?]"
  fi
}

echo "硬件信息采集报告"
echo "生成时间: $(date '+%Y-%m-%d %H:%M:%S %z')"
echo "主机名  : $(hostname)"
echo "输出文件: $OUT"
echo "原始表  : $RAW/"

sec "0. 系统与内核"
run "内核" "uname -a"
run "发行版" "cat /etc/os-release 2>/dev/null | head -8"
run "启动参数" "cat /proc/cmdline"
run "页大小" "getconf PAGE_SIZE"
run "小页/大页" "grep -E 'HugePages_Total|Hugepagesize|AnonHugePages|MemTotal|MemFree|MemAvailable|SwapTotal' /proc/meminfo"
run "THP 策略" "cat /sys/kernel/mm/transparent_hugepage/enabled 2>/dev/null"

sec "1. CPU"
run "lscpu" "lscpu"
run "缓存（lscpu -C）" "lscpu -C"
run "每核 topology（前 40 行）" "lscpu -e | head -40"
run "cpu0 的缓存 sysfs" "for i in /sys/devices/system/cpu/cpu0/cache/index*; do echo \"[\$i]\"; for f in level type size ways_of_associativity coherency_line_size shared_cpu_list; do echo -n \"  \$f=\"; cat \$i/\$f 2>/dev/null; done; done"
run "CPU part / 型号" "grep -m1 -E 'CPU implementer|CPU part|CPU architecture' /proc/cpuinfo; echo; grep -m1 'Features' /proc/cpuinfo"

sec "2. NUMA 拓扑"
run "numactl -H（概览）" "numactl -H | head -40"
run "每个 node 的 CPU 列表" "for n in /sys/devices/system/node/node*; do echo -n \"\$(basename \$n): \"; cat \$n/cpulist 2>/dev/null; echo; done"
run "每个 node 的内存总量" "for n in /sys/devices/system/node/node*; do echo -n \"\$(basename \$n): \"; grep MemTotal \$n/meminfo 2>/dev/null; done"
run "node5 的距离表（前 32 项）" "cat /sys/devices/system/node/node5/distance 2>/dev/null"
run "node16 的距离表（前 32 项）" "cat /sys/devices/system/node/node16/distance 2>/dev/null"
run "lstopo 结构（前 80 行）" "lstopo-no-graphics --of console 2>/dev/null | head -80"
run "hwloc 对象统计" "hwloc-info 2>/dev/null | head -20"

sec "3. 内存条（DMI，最关键）"
run "物理内存阵列（dmidecode -t 16）" "dmidecode -t 16"
run "每条内存（dmidecode -t 17）" "dmidecode -t 17"
run "内存概览（dmidecode -t memory）" "dmidecode -t memory"
run "缓存（dmidecode -t cache）" "dmidecode -t cache"

sec "4. ACPI：NUMA/内存拓扑的固件描述"
for t in SRAT SLIT HMAT NFIT; do
  if [ -e "/sys/firmware/acpi/tables/$t" ]; then
    cp "/sys/firmware/acpi/tables/$t" "$RAW/$t.bin" 2>/dev/null
    printf '\n----- %s（字符串视图，前 60 行） -----\n' "$t"
    echo "原始表已存: $RAW/$t.bin（$(stat -c%s "$RAW/$t.bin" 2>/dev/null) 字节）"
    strings "$RAW/$t.bin" 2>/dev/null | head -60
  else
    printf '\n----- %s -----\n不存在（这台机器的固件没暴露）\n' "$t"
  fi
done

sec "5. 内核怎么看 NUMA / 内存"
run "dmesg 里的 NUMA/SRAT/HMAT" "dmesg 2>/dev/null | grep -iE 'numa|srat|hmat|slit|node.*mem' | head -60"
run "dmesg 里的 EDAC / 内存控制器" "dmesg 2>/dev/null | grep -iE 'edac|ddrc|memory controller|hbm|cxl' | head -40"
run "EDAC 实例" "ls -l /sys/devices/system/edac/mc/ 2>/dev/null"

sec "6. 物理地址布局"
run "/proc/iomem（前 60 行）" "head -60 /proc/iomem"

sec "7. 内存通道计数器（华为 DDRC PMU）"
run "DDRC 设备数量与清单" "ls /sys/bus/event_source/devices/ | grep -c ddrc; ls /sys/bus/event_source/devices/ | grep ddrc"
run "单个 DDRC 暴露的事件" "for e in /sys/bus/event_source/devices/hisi_sccl0_ddrc0_0/events/*; do echo -n \"\$(basename \$e): \"; cat \$e; done 2>/dev/null"
run "空载 1 秒的通道读数（背景噪声）" "EV=\$(for s in 0 1 8 9; do for d in 0 1 2 3; do for c in 0 1; do echo -n \"hisi_sccl\${s}_ddrc\${d}_\${c}/flux_rd/,\"; done; done; done | sed 's/,\$//'); perf stat -a -e \"\$EV\" sleep 1 2>&1 | tail -40"

sec "8. 内存使用现状"
run "numastat -m" "numastat -m 2>&1 | head -30"
run "每 node 的 zone 概要" "awk '/^Node [0-9]+, zone/{n=\$2; z=\$4} /managed/{if(\$1==\"managed\") printf \"node%-3s %-8s managed=%s pages\\n\", n, z, \$2}' /proc/zoneinfo | head -60"

sec "9. 其他"
run "大页配置" "cat /proc/sys/vm/nr_hugepages 2>/dev/null; cat /sys/kernel/mm/transparent_hugepage/defrag 2>/dev/null"
run "当前负载" "uptime; cat /proc/loadavg"

if [ "$REDACT" -eq 1 ] && [ -f "$OUT" ]; then
  sed -i -E 's/^(\s*(Serial Number|Asset Tag|UUID|SKU Number|Part Number):).*/\1 [REDACTED]/I' "$OUT"
  echo
  echo "（已按 --redact 去掉序列号/资产标签/UUID/料号）"
fi

echo
echo "============================================================"
echo "完成：$OUT"
echo "行数：$(wc -l < "$OUT")"
echo "============================================================"

# 把摘要打到终端（报告本体在文件里；不能用 tee，会截断文件）
SUMMARY=$(
  echo
  echo "===== 快速摘要 ====="
  grep -m1 "Model name" "$OUT" 2>/dev/null
  awk '/^CPU\(s\):|^Thread\(s\) per core:|^Core\(s\) per socket:|^Socket\(s\):|^NUMA node\(s\):/{print}' "$OUT" | head -6
  echo -n "DDRC 通道实例数: "; ls /sys/bus/event_source/devices/ 2>/dev/null | grep -c ddrc
  echo -n "内存条条目数: "; grep -c "^Memory Device$" "$OUT" 2>/dev/null
)
printf '%s\n' "$SUMMARY"
printf '%s\n' "$SUMMARY" >&3
