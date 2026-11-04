# 这台机器的内存硬件：从通道到跨 NUMA，一次掰碎的实测

> **这份文档是个人学习向的**：不服务于本仓的任何优化决策，只回答"这台机器的内存到底
> 长什么样、怎么用起来、能跑多快"。**硬件规格来自 `dmidecode`（root 采集，脱敏全文在
> `docs/hw-report-20261104-144311.txt`），性能数字是 2026-11-04 在这台机器上现测的**。
> 命令速查在 §8、完整调查顺序在附录 B、探针源码在附录 A。
>
> 如果你要的是"本仓的性能分析用哪个数"，那是另一条线；本文是纯硬件向的"为什么"和
> "怎么自己测"。另有一份同类的实测台账 `memory-subsystem.md`（延迟阶梯、L2 容量、
> THP 等），两份互不依赖，按兴趣看。
>
> **一句话预告结论**：这台机器插了 **16 条 64GB 的 DDR5-5600**（ECC、双 rank），
> 分在 **16 条内存通道**（每 socket 8 条、每 SCCL 4 条）。DDR5 把每条通道再拆成
> 两个 32 位子通道，所以内核里能数出 **32 个计数器**。单核最多拉 **19 GB/s**，一个
> NUMA node（38 核）就能把 4 条通道跑满（**168 GB/s**），整机 **658 GB/s**。跨 NUMA
> 每多一跳，延迟约 +50~130 ns、带宽掉一档；跨 socket 的单 node 只能拿到 **17 GB/s**。
> 另外还有 **64GB 的"封装内内存"**（每颗 CPU 32GB，10000 MT/s），被切成 16 个 4GB
> 的纯内存 NUMA node。

---

## 0. 一页纸：先看这 15 个数

| # | 是什么 | 实测值 | 怎么来的 |
|---|---|---|---|
| 1 | CPU / NUMA | 2 socket × 304 核 = **608 核**；16 个带 CPU 的 node（各 38 核）+ 16 个无 CPU 的封装内内存 node（各 4GB） | `lscpu`、`numactl -H` |
| 2 | 内存条 | **16 × 64GB DDR5-5600**，双 rank，Hynix，带 ECC（80 位 = 64 数据 + 8 校验） | `dmidecode -t 17`（root） |
| 3 | 插法 | 每 socket 8 条，`SOCKET 0/1 CHANNEL 0..7 DIMM 0`——**一条通道插一条，16 个槽全满** | 同上 |
| 4 | 内存通道总数 | **16 条**（4 SCCL × 4 条 / 每 socket 8 条）；DDR5 再拆成 32 个 32 位子通道 | `dmidecode` + PMU 设备名 |
| 5 | 每个 SCCL | **4 条通道（8 个子通道）**，服务 4 个 NUMA node | 实测映射（§2.4） |
| 6 | DDR 控制器时钟 | **1405 MHz** = 数据率 ÷ 4（5600/4） | `perf stat -e .../cycles/` |
| 7 | 每通道带宽 | **44.8 GB/s**（8 字节 × 5600 MT/s）；实测 42.1 | 规格 vs 实测 |
| 8 | 单核顺序读 | **19.0 GB/s** | `memprobe bandwidth 1024 1 rd` |
| 9 | 单 NUMA node（38 核） | **168 GB/s**（16 核就饱和，= 4 条通道的 94%） | 同上，16/24/38 线程 |
| 10 | 单 SCCL（4 node，152 核） | **171 GB/s**（= 同一组 4 条通道） | 时间窗口法（附录 A） |
| 11 | 整机（4 SCCL） | **658 GB/s**（理论 716.8，92%） | 时间窗口法 |
| 12 | 本地 DRAM 延迟 | **128-148 ns** | `memprobe chase 64MB` |
| 13 | NUMA 每跳代价 | 距离 10/20/30/40 → **130 / 180 / 255 / 355 ns**；封装内内存（61/71/81/91）→ **141-447 ns** | 核固定 node5，只改内存位置 |
| 14 | 随机 vs 顺序（单核） | 6.8 vs 19.0 GB/s（**2.8x**） | `gathermt rand` vs `bandwidth` |
| 15 | 封装内内存带宽 | 单窗口 **378-410 GB/s**（DRAM node 的 2.3 倍）、十四个窗口 **5238 GB/s**；跨 SCCL 是 **173 GB/s** 的链路墙、跨 socket 只剩 19 | 配对核 + `memwin`（§6.5） |

> 第 6、7 行是本文最"硬"的发现：**不需要 root，用 perf 读 DDR 控制器的周期数，
> 就能反推出内存跑在多快**。而第 2-4 行反过来证明了一件事：**反推只能给数量级，
> 代际和型号必须靠 `dmidecode`**——我一开始就从时钟反推成"DDR4-28xx"，被真实数据
> 打脸了（真实是 DDR5-5600）。这个教训写在 §2.4。

---

## 1. 先建词汇：内存不是"一大块"

### 1.1 一个类比：图书馆

把内存想成一座图书馆：

| 计算机术语 | 图书馆里的东西 | 本机实测 |
|---|---|---|
| **通道（channel）** | 通往书库的**走廊** | 16 条（每 socket 8 条），每条 44.8 GB/s |
| **子通道（sub-channel）** | 走廊里的**两条并排通道** | 32 个，每条 22.4 GB/s（DDR5 特有） |
| **内存控制器（DDRC）** | 每条走廊的**管理员** | 每 SCCL 4 个，每个管 1 条通道（2 个子通道） |
| **DIMM** | 一排**书架** | 16 排，每排 64GB |
| **rank** | 书架的一**层** | 每条 2 层（双 rank） |
| **bank** | 一层里的**一格** | 每格可以独立打开 |
| **row（行）** | 一格里的**一行书** | 打开要 ~13-15ns（row activate） |
| **column（列）** | 行里的**某一本** | 取一本很快 |
| **cacheline** | 你一次**借走的 64 字节** | 本机 64B |

**最关键的一条**：打开"一行书"（row activate）很贵，但同一行里连续取很多本很便宜。
这就是为什么**顺序访问比随机访问快**——不是"顺序"本身有魔力，而是它让每一行书
只被翻开一次。

### 1.2 再补三个词

- **NUMA**（Non-Uniform Memory Access）：内存不是"离所有核一样远"。每个核访问
  "自己家"的内存快，访问别人家的慢。本机有 32 个 NUMA node。
- **SCCL**：这是本机（华为鲲鹏系）内部的"超级簇"叫法。**一个 SCCL 有 4 条内存通道
  （8 个子通道），服务 4 个 NUMA node**。你可以把它理解成"一个内存子站"。
  > 注意：SCCL 不是教科书词，是这台机器的固件/内核叫法（PMU 名字里带 `sccl`）。
- **socket（路）**：CPU 插槽。本机 2 路，每路 304 核、2 个 SCCL、8 条内存通道。

### 1.3 一次访问的完整路径

```
核心
 └─ L1d（32KB，2.0ns）
     └─ L2（768KB，7.4-8.8ns）
         └─ 【本机没有 L3】
             └─ SCCL 的 HHA（Home Agent，管"这条 cacheline 归谁"）
                 └─ DDR 控制器（DDRC）
                     └─ 通道 → DIMM → rank → bank → row → column
                         └─ 本地 128-148ns；跨 node 180-355ns
```

**"没有 L3"这件事很重要**：数据要么在 L2（几个 ns），要么直接付 128ns。中间没有
缓冲层。所以本机的性能曲线特别"两极"。

### 1.4 再往下挖：DDR 内存条内部到底在干什么

前面把 DRAM 说成"书架"，这一节把它拆到能看见"机械动作"的层级。看不懂可以跳过，
不影响后面的实测结论。

**一条内存条（DIMM）里有什么**

```
┌─────────────────────────────────────────────┐
│ DIMM（一块绿色 PCB）                         │
│  ├─ DRAM 颗粒 × N（每颗 8/16/32 bit 宽）     │
│  ├─ SPD 小芯片（存"我是谁、多快、多少容量"） │
│  ├─ RCD（服务器 RDIMM 才有：命令/地址缓冲）  │
│  └─ ECC 校验位（服务器通常 64+8 bit）        │
└─────────────────────────────────────────────┘
```

- **rank**：把若干颗粒拼成 **64 位宽**，就叫一个 rank。所以"双 rank"= 两组颗粒，
  同一时刻只有一组在数据线上。本机的内存条是 **2 rank**。
- **bank**：rank 内部再切成多个"存储阵列"，每个 bank 有一个**行缓冲（row buffer）**。
  DDR5 通常每 rank **32 个 bank**（8 个 bank group × 4 个 bank），比 DDR4 多一倍。
- **row / column**：一个 bank 是二维阵列；`activate` 打开一行到 row buffer，
  之后 `read/write` 只给列地址。
- **子通道（sub-channel）**：DDR5 才有的东西。一条内存条的 64 位数据线被拆成
  **两个独立的 32 位子通道**，各有自己的命令总线，可以同时干两件事。这是 DDR5
  同频率下比 DDR4 快的核心原因，也是本文"32 个计数器 vs 16 条通道"的来源。

**一次访问的三个动作（这是随机慢的根源）**

| 动作 | 干什么 | 典型耗时 | 什么时候发生 |
|---|---|---|---|
| `ACT`（activate） | 把一整行搬进 row buffer | **~13-15 ns** | 换行时 |
| `RD/WR`（列访问） | 从 row buffer 取/写一段 | ~7-8 ns（CL） | 每次 |
| `PRE`（precharge） | 关闭当前行、准备开下一行 | **~14 ns** | 再换行时 |

- **row buffer hit**：数据就在当前打开的行里 → 只付列访问，很快。
- **row buffer miss**：要 `PRE` + `ACT` + `RD` → 多付 ~27ns。

**这就是"顺序比随机快"的物理原因**：顺序访问把同一行里的数据连着取完（大量 hit），
随机访问每取一次就换一行（每次 miss）。实测单核顺序 19.0 GB/s、随机 6.8 GB/s，
差的 2.8 倍主要来自这里，不是"预取器"一个因素。

**一次传输多少字节**

DDR5 的突发长度是 16（BL16），但它是"每 32 位子通道"算的：**32 位 × 16 拍 = 64 字节**，
正好一条 cacheline。所以"取一条 cacheline"在物理上就是**两个子通道各出 32 字节**。

这也解释了控制器计数器的奇怪单位：本文实测 1 秒读 166.4 GB，对应 5212.9M 条读命令
= **31.9 字节/条**。因为**每个 `flux_rd` 计数器只负责一个 32 位子通道**，一次记账
就是半个 cacheline。

**刷新（refresh）：内存永远跑不到 100%**

DRAM 的电容会漏电，必须周期性刷新。JEDEC 规定**每 64ms 把每一行刷新一遍**
（高温时缩短到 32ms）。分摊下来每个 bank 每隔几微秒就要刷一次，刷新期间那个 bank
不能服务请求。

这解释了一个实测现象：本机单 SCCL 理论 **179.2 GB/s**，实测最好 **171 GB/s（95%）**；
整机理论 **716.8 GB/s**，实测 **658 GB/s（92%）**。**差的 5-8% 不是代码问题，是刷新、
ECC、命令开销这些物理成本。**

**ECC 与 RDIMM：服务器内存为什么"更稳但略慢"**

- **ECC**：每 64 位数据加 8 位校验。本机 `dmidecode` 就写着 `Total Width: 80 bits /
  Data Width: 64 bits`——**多出来的 16 位就是校验位**（每条通道 8 位）。它能纠正
  1 bit 错、发现 2 bit 错。代价是**部分写要"读-改-写"**（RMW）：只写半个 cacheline
  时，得先把整条读进来算校验。
- **RDIMM**：命令/地址先过一个缓冲芯片（RCD），让一条内存条能挂更多 rank、更大容量，
  代价是多 1-2 拍延迟。

这两样在服务器上是默认的，也是"为什么服务器内存延迟 128ns，而桌面机可以做到
80-90ns"的一部分原因。

---

## 2. 这台机器长什么样

### 2.1 拓扑（`lscpu` + `numactl -H` + `lstopo`）

```
Package 0 (socket 0)                      Package 1 (socket 1)
├─ SCCL 0  → NUMA node 0,1,2,3  (各38核)   ├─ SCCL 8  → node 8,9,10,11
├─ SCCL 1  → NUMA node 4,5,6,7  (各38核)   ├─ SCCL 9  → node 12,13,14,15
├─ 8 条 DDR5 通道（4+4）                   ├─ 8 条 DDR5 通道（4+4）
└─ 32GB 封装内内存                          └─ 32GB 封装内内存
      ↓ 切成 8 个 4GB node                       ↓ 切成 8 个 4GB node
   node 16-23（无 CPU）                       node 24-31（无 CPU）
```

几个容易搞混的点：

- `lscpu` 说 **2 socket、每 socket 304 核**，但 `numactl` 说 **32 个 node**。
  两个都对：**socket 是插槽，node 是"离哪块内存近"的分组**，粒度更细。
- 每 node **38 核**，不是整数幂——这是硬件布线决定的，别去凑。
- **没有 SMT**（`Thread(s) per core: 1`）：608 核 = 608 线程，没有超线程。
- 全文的"XX GB/s"都是 **10 进制 GB**（1e9 B/s），与 `memprobe` 的输出口径一致；
  内存容量用 GiB 会更小一点，看到两个数不一致别怀疑。

### 2.2 内存条本身：16 × 64GB DDR5-5600，全插满

这一段必须要 root——`dmidecode` 读的是 BIOS 的 SMBIOS 表，是最权威的硬件自述。

```bash
$ sudo dmidecode -t 17 | grep -E "Size|Locator|Type:|Speed:|Rank|Width"
        Total Width: 80 bits
        Data Width: 64 bits
        Size: 64 GB
        Locator: DIMM000
        Bank Locator: SOCKET 0 CHANNEL 0 DIMM 0
        Type: DDR5
        Speed: 5600 MT/s
        Rank: 2
        Configured Memory Speed: 5600 MT/s     ← 没有降频
```

16 条完全一样，插法是：

| socket | 通道 | 条数 | 单条容量 |
|---|---|---|---|
| 0 | CHANNEL 0,1,2,3,4,5,6,7 | 8 | 64 GB |
| 1 | CHANNEL 0,1,2,3,4,5,6,7 | 8 | 64 GB |

**一条通道插一条、16 个槽全满、没有空格**。加上阵列信息：

| 项 | 值 | 含义 |
|---|---|---|
| 内存阵列总容量上限 | **4 TB** | 主板能插到 4TB（现在是 1TB） |
| 阵列里的设备数 | 16 | 16 个 DIMM 槽，全占了 |
| 错误纠正类型 | **Single-bit ECC** | 单比特纠错 |
| 单条 | 64 GB / DDR5 / 5600 MT/s / 2 rank / Hynix | 型号 `HMCG94AGBRA181N` |
| 位宽 | **80 位 = 64 数据 + 16 校验** | 服务器 ECC 的标准形态 |

整机 `MemTotal` 是 **1121975468 kB ≈ 1.045 TiB**。16 × 64 GiB = 1024 GiB，加上
封装内内存 64 GiB，正好约 1070 GiB——比 1088 GiB 少的那十几 GB 是固件/内核预留。

> **"80 位宽"这个数字值得单独记一下**：它是判断"这条通道到底带不带 ECC"最直接的
> 证据，比看 `Type: DDR5` 有用。非 ECC 的机器这里会是 `64 bits / 64 bits`。

`dmidecode` 那几行字段，第一次看容易懵，逐个翻译一下：

| 字段 | 中文 | 怎么用 |
|---|---|---|
| `Size` | 单条容量 | 16 × 64 GB = 1 TB，和 `MemTotal` 对得上就说明没有半条没插好 |
| `Locator` | 槽位丝印名 | 如 `DIMM000`，是主板上的丝印，拆机时按它找 |
| `Bank Locator` | **插在哪个 socket 的哪条通道** | `SOCKET 0 CHANNEL 3 DIMM 0`——**判断通道数的关键** |
| `Type` | 内存代际 | `DDR5` |
| `Speed` | 内存条额定速率 | `5600 MT/s`（这条内存条能跑多快） |
| `Configured Memory Speed` | **实际跑的速率** | 也是 5600 → **没有被 BIOS 降频** |
| `Rank` | 内存条有几组颗粒 | `2` = 双 rank，同一时刻只有一组占数据线 |
| `Total Width` / `Data Width` | 总位宽 / 数据位宽 | `80/64` = 带 ECC；`64/64` = 不带 |
| `Form Factor` | 封装形态 | `DIMM` = 插槽内存；`Die` = 直接封装在 CPU 上（见 §6.5） |
| `Array Handle` | 属于哪个内存阵列 | `0xFFFE` 表示"不归主板内存阵列管" |

> `Speed` 和 `Configured Memory Speed` 不一致，就是被降频了（BIOS 保守设置、混插
> 不同规格、或者通道数太多跑不到额定）。**这两个数不一致时，以实测为准。**

### 2.3 16 条通道 vs 32 个计数器：DDR5 的子通道陷阱

这是全文最容易搞错的一处，我自己就先搞错过。

内核把每个 DDR 控制器实例都暴露成一个 perf PMU 设备，数出来是 32 个：

```bash
$ ls /sys/bus/event_source/devices/ | grep ddrc | wc -l
32
$ ls /sys/bus/event_source/devices/ | grep ddrc | head -4
hisi_sccl0_ddrc0_0
hisi_sccl0_ddrc0_1
hisi_sccl0_ddrc1_0
hisi_sccl0_ddrc1_1
```

命名是 `hisi_sccl<簇>_ddrc<控制器>_<子通道>`。**但 32 不是通道数**——`dmidecode`
明明白白只有 16 条 DIMM、插在 16 条通道上。对不上的原因是 DDR5：

```
1 条 DDR5 通道（64 位数据线）
 ├─ 子通道 0（32 位）→ 一个 ddrc 实例 _0
 └─ 子通道 1（32 位）→ 一个 ddrc 实例 _1
```

所以数量关系是：

```
16 条通道 × 2 个子通道 = 32 个 PMU 实例
每个 SCCL = 4 条通道 × 2 = 8 个实例
每个 socket = 8 条通道 × 2 = 16 个实例
```

**为什么这个区别重要**：把 32 当成"32 条通道"再去除以带宽，每通道就只有 22.4 GB/s，
会得出"DDR4-2800"的错误结论（我一开始就是这么错的）。实际每条通道是 **44.8 GB/s**。

每个子通道带三个计数器（**不需要 root**，本机 `perf_event_paranoid = -1`）：

| 事件 | 含义 |
|---|---|
| `cycles` | 这个子通道的时钟周期数（1405 MHz） |
| `flux_rd` | 读命令数（实测一条 ≈ 32 字节 = 半个 cacheline） |
| `flux_wr` | 写命令数 |

### 2.4 哪个 NUMA node 用哪 4 条通道（实测）

方法：把一个进程绑在某个 node 的核上、内存也绑到同一个 node，读 1GB，看哪些
计数器在涨。

| 跑的 node | 主要发力的 SCCL | 占比 |
|---|---|---|
| node 0、1 | **sccl0** | 77-82% |
| node 4、5、6、7 | **sccl1** | 76-82% |
| node 8 | **sccl8** | 79% |
| node 12 | **sccl9** | 95% |

结论：**每 4 个 NUMA node 共享一个 SCCL 的 4 条通道（8 个子通道）**。

```
node 0 ┐
node 1 ├─ 共享 sccl0 的 4 条通道 ── 合计上限 ~179 GB/s（实测 171）
node 2 │
node 3 ┘
```

> **这条改变了你对"NUMA node"的直觉**：node 5 和 node 6 虽然名字不同、延迟也不同
> （130 vs 180ns），但它们**用的是同一组内存通道**。所以"多用一个 node"不等于
> "多一份带宽"——只有跨 SCCL 才加带宽。

**顺便纠一个测量陷阱**：跑 node5 时会看到 sccl9 也有 **14-20%** 的流量。别以为这是
"内存串了"——空载 1 秒 sccl9 本身就有 **~21M** 条读命令（其他租户/后台），这个比例
会随运行时段明显波动。**判据不是"哪个 SCCL 有数"，而是"目标 SCCL 的 8 个计数器是否
严格均匀"**：node5 的 8 个一直在 4.39-4.42M 之间，误差 < 1%，这才是本地流量的指纹。
**用计数器必须先量空载背景。**

### 2.5 反推内存频率：不用 root 也能测，但只能给量级

`cycles` 计数器一秒涨多少，就是控制器的时钟：

```bash
$ perf stat -a -e hisi_sccl1_ddrc0_0/cycles/ sleep 1
   1,405,xxx,xxx   hisi_sccl1_ddrc0_0/cycles/
→ 1405 MHz
```

**DDR5 的控制器时钟 = 数据率 ÷ 4**（DDR5 内部是四相接口，不是 DDR4 的两倍关系）：

```
1405 MHz × 4 ≈ 5600 MT/s          ← 正好是 dmidecode 说的 DDR5-5600
```

带宽随之而来：

```
每个子通道 = 4 字节 × 5600 MT/s = 22.4 GB/s
每条通道   = 2 个子通道         = 44.8 GB/s
每 SCCL    = 4 条通道           = 179.2 GB/s   → 实测 171（95%）
每 socket  = 8 条通道           = 358.4 GB/s   → 实测 339（95%）
整机       = 16 条通道          = 716.8 GB/s   → 实测 658（92%）
```

**这里有个必须交代的错误**：我第一次把 32 个实例当成了 32 条通道，于是按
"一条通道 64 位（8 字节）" × "DDR 双倍数据率（2 × 1405 = 2810 MT/s）"算出
每通道 22.5 GB/s，顺手把 2810 MT/s 翻译成"DDR4-2800/2933"。
真实答案是 **DDR5-5600**——我同时错了两处：**实例是 32 位而非 64 位**（少算一半）、
**DDR5 控制器时钟是 ÷4 而非 ÷2**（少算一半）。有意思的是，两个错误在"每通道带宽"
这个数上恰好互相抵消，所以推算出来的 22.5 GB/s 看起来和实测 21 GB/s"吻合"，
错误结论因此显得非常自洽。

> **方法论教训**：反推出来的频率只能告诉你"大概什么量级"，**代际（DDR4/DDR5）、
> 型号、容量、rank 数，必须让固件自己说**（`dmidecode`）。任何"从性能数字反推
> 硬件型号"的结论，都要标注为待验证。

---

## 3. 内存是怎么被"用上"的

### 3.1 从 `malloc` 到 DRAM：中间发生了什么

1. **`malloc` 只给虚拟地址**，不碰物理内存。
2. 第一次**写**这个地址 → 触发缺页 → 内核分配一个物理页（4KB 或 2MB 大页）。
3. **物理页落在哪个 NUMA node，取决于"第一次碰它的那个线程当时在哪颗核上"**
   ——这就是 first touch。它决定了你后面是 130ns 还是 255ns。
4. 之后每次访问：虚拟地址 → TLB → 物理地址 → 地址交织 → 通道/rank/bank。

**所以"内存是怎么用上的"第一课**：不是 `malloc` 决定的，是**第一次写它的那颗核**
决定的。多线程程序如果只在主线程初始化大数组，整块内存都会落在主线程那个 node 上。

### 3.2 地址交织：一条顺序流会自动铺满 8 个子通道

硬件会把**连续的物理地址轮流分给不同通道**（粒度通常是 256B-1KB）。所以哪怕只有
一个核在顺序读，一个 SCCL 的 4 条通道（8 个子通道）也会一起干活。实测
（node5，单核读 1GB，单位是"读命令数"）：

| 子通道 | ddrc0_0 | ddrc0_1 | ddrc1_0 | ddrc1_1 | ddrc2_0 | ddrc2_1 | ddrc3_0 | ddrc3_1 |
|---|---|---|---|---|---|---|---|---|
| 读命令 | 4.40M | 4.40M | 4.39M | 4.38M | 4.39M | 4.39M | 4.39M | 4.39M |

**8 个子通道误差 < 1%**。38 线程时也一样（4.40-4.42M）。

这条的工程含义：**你不需要为了"用满带宽"去手动把数据分到不同通道**——硬件已经做了。
你需要关心的是"够不够多核"和"访问是不是随机"。

### 3.3 64 字节的 cacheline 与 RFO

- CPU 从内存取数的**最小单位是 64 字节**（一条 cacheline），哪怕你只要 4 字节。
- **写**的时候更贵：如果这条 line 不在缓存里，CPU 要先把它**读进来**再改，这叫
  **RFO（Read For Ownership）**。所以"写 1 字节"在缓存没命中时可能是"读 64B + 写 64B"。
- 这也解释了本机实测的一个现象：**拷贝（读+写）在 38 线程时反而比 16 线程慢**
  （145 → 122 GB/s）——写流量把通道和设备队列挤爆了。

### 3.4 顺序 vs 随机：差 2.8 倍，而且越并行差得越多

| 线程 | 顺序流（GB/s） | 随机 64B 取数（GB/s） | 差距 |
|---|---|---|---|
| 1 | 19.0 | 6.8 | 2.8x |
| 8 | 128.6 | 35.2 | 3.7x |
| 38 | 168.3 | 42.1 | **4.0x** |

两个原因叠加：

1. **row buffer**：顺序读时同一行书只翻开一次；随机读每本书都要重新翻行。
2. **预取器**：硬件看到"地址在递增"就会提前取后面的 line。随机地址它完全帮不上忙。

**而且随机访问在高线程下会变差**：38 个线程各读各的随机地址，38 条流互相打断
对方的 row，每个访问的延迟从单核的 9.4ns 涨到 57.8ns。**随机访问不是"多核就能
线性扩展"的**——随机行取数（`x[idx]` 这类负载）的核心特征。

### 3.5 大页与 TLB：为什么"4KB 页"也有成本

虚拟地址转物理地址要查页表。为加速，CPU 里有 **TLB**（Translation Lookaside
Buffer，可理解为"最近用过的页表项缓存"）。TLB 项数有限，而每个 4KB 页只覆盖 4KB：

| 页大小 | 覆盖 1GB 需要多少项 | TLB 能装下吗 |
|---|---|---|
| 4KB | 262,144 项 | **装不下** → 频繁 miss |
| 2MB（大页） | 512 项 | 轻松装下 |
| 1GB（超大页） | 1 项 | 1 项 |

TLB miss 后要走多级页表（最坏 4 次内存访问），看起来是个大问题。**但本机实测：
开/关 2MB 大页（THP），gather 的性能没有差别。**

为什么？因为 **page walk 和数据 miss 是并行发生的**：数据 miss 本来就要等 128ns，
页表的那几次访问被这段等待"吸收"掉了。

> **一条通用经验**：一个延迟如果**藏在更长的延迟后面**，优化它就是 0 收益。
> 判断方法是"关掉它，看总时间动不动"，不是"算它能省多少"。
>
> 大页真正有用的是**另一个场景**：工作集巨大、访问分散、而且 TLB 容量真的成为
> 瓶颈时（比如 100GB 级随机访问）。本机这种"128ns 主宰一切"的形态，轮不到 TLB。

---

## 4. 带宽实测：从小到大的六级台阶

统一口径：读、1GB 缓冲区（远大于 L2）、顺序、绑核 + 绑内存到 node5。

| 规模 | 核数 | 实测读带宽 | 用了几条通道 | 说明 |
|---|---|---|---|---|
| 单核 | 1 | **19.0 GB/s** | 1（沾了一半） | 一个核的极限，只有一条通道的 42% |
| 几核 | 2 | 37.8 | 1 | 几乎线性 |
| | 4 | 73.0 | 1-2 | 几乎线性 |
| | 8 | 128.6 | 3-4 | 开始接近上限 |
| | 16 | **164.7** | 4 | **已经饱和** |
| | 24 | 169.4 | 4 | 再多人也没用 |
| 单 NUMA node | 38 | 168.3 | 4 | 4 通道（8 子通道）跑满 |
| 单 SCCL | 152（4 node） | **171.3** | 4 | 和单 node 一样——**同一组通道**；理论 179.2，95% |
| 单 socket | 304（2 SCCL） | **339.3** | 8 | ×2；理论 358.4，95% |
| 整机 | 608（4 SCCL） | **658.2** | 16 | ×4；理论 716.8，效率 92% |

### 4.1 三个可以直接用的结论

1. **单核最多 19 GB/s**。哪怕内存通道空着，一个核也拉不动更多——它受限于自己的
   在飞请求数（MSHR）。注意 19 GB/s **连一条 DDR5 通道（44.8 GB/s）都喂不满**，
   只有 42%。所以"单线程跑内存密集任务"永远是在浪费通道。
2. **16 核就能把一个 node 的 4 条通道跑满**。38 核的 node 跑内存密集任务，后 22 核
   基本白给。**别盲目开满线程。**
3. **加 node 不一定加带宽**：node5→node6 是同一组通道，带宽不变；只有跨 SCCL
   才翻倍。整机要跑到 658 GB/s，必须四个 SCCL 一起上。

### 4.2 拷贝为什么比读更"吃"通道

| 线程 | 读 GB/s | 拷贝 GB/s（读+写合计） |
|---|---|---|
| 1 | 19.0 | 20.3 |
| 8 | 128.6 | 137.2 |
| 16 | 164.7 | 145.1 |
| 38 | 168.3 | **121.9** |

拷贝的"合计字节"看起来更高，但**每搬一个字节要付读+写两份通道流量**，而且写会
触发 RFO。线程一多，写流量开始互相排队，总吞吐反而掉。

---

## 5. 延迟实测：从 2ns 到 355ns

| 层级 | 工作集 | 实测延迟 | 备注 |
|---|---|---|---|
| L1 命中 | 4-32KB | **2.0 ns** | 4 拍 |
| L2 命中 | 256-576KB | **7.4-8.8 ns** | 有效容量约 640-704KB |
| L2 装不下 | 768KB-8MB | 17-95 ns | 直接进 DRAM，但还能蹭到 row buffer |
| 本地 DRAM | 64-128MB | **128-148 ns** | 这就是"一次随机取数"的价格 |
| 同 SCCL 的邻 node | — | **170-191 ns** | 距离 20 |
| 同 socket 另一 SCCL | — | **255 ns** | 距离 30 |
| 跨 socket | — | **354-387 ns** | 距离 40 |
| 那 16 个 4GB 的封装内内存 node | — | **141-285 ns** | 距离 61/71/81 三档，§6.5 |

**NUMA 距离不是抽象单位**：本机距离每 +10，实测延迟大约 +50~130ns。距离表在
`/sys/devices/system/node/node5/distance`，第一行就是 node0..node31 的距离。

---

## 6. 跨 NUMA 的细节

### 6.1 延迟：只改"内存放哪"，核不动

核固定在 node5（CPU 200），用 `--membind` 换内存位置：

| 内存所在 | 距离 | 1 线程读 | 8 线程 | 38 线程 |
|---|---|---|---|---|
| node5（本地） | 10 | **19.1 GB/s** | 128.7 | 163.3 |
| node6（同 SCCL） | 20 | 9.1 | 64.9 | 149.3 |
| node0（同 socket，另一 SCCL） | 30 | 6.4 | 47.8 | 127.8 |
| node8（跨 socket） | 40 | **4.4** | **17.2** | **17.2** |

三条读法：

1. **低线程看延迟，高线程看带宽**。1 线程时跨 node 直接腰斩（19→9→6→4.4）；
   38 线程时因为本地核已经把通道压满，差距反而没那么夸张（163→149→128）。
2. **跨 socket 是断崖**：38 线程也只有 **17 GB/s**，是本地的一个零头。**跨 socket
   的互连不是给你跑大流量的**，它是给缓存一致性和小消息用的。
3. **同 SCCL 的两个 node 延迟差 50ns，但带宽上限一样**——因为共用 4 条通道。
   延迟差来自片内互连的跳数。

### 6.2 距离表怎么读

```bash
$ cat /sys/devices/system/node/node5/distance
30 30 30 30  20 10 20 20  40 40 40 40 40 40 40 40  81 81 81 81 71 61 71 71  91 91 91 91 91 91 91 91
└── node0-3 ┘└─ node4-7 ─┘└────── node8-15 ──────┘└────── node16-23 ──────┘└────── node24-31 ──────┘
   同socket      自己所在          跨 socket            CPU1 封装内              CPU2 封装内
   另一 SCCL     的 SCCL
```

- `10` = 自己（本地）
- `20` = 同一个 SCCL 的兄弟 node（一跳，node4/6/7）
- `30` = 同 socket、另一个 SCCL（node0-3）
- `40` = 跨 socket（node8-15）
- `61-91` = 那 16 个 4GB 的封装内内存 node（§6.5）。**规律已经查清**（解析 SLIT
  原表得到）：每个 CPU node 与一个封装内 node **1:1 配对**，`node i ↔ node 16+i`
  是 **61**（最近）；同 SCCL 的另外三个是 **71**；同 socket 另一个 SCCL 的四个是
  **81**；另一个 socket 的八个是 **91**。四组正好对应四个 SCCL：
  node0-3↔16-19、4-7↔20-23、8-11↔24-27、12-15↔28-31。
- `255` = 不可直达。**node16-31 彼此之间全是 255**（除了自己），它们只能通过
  带 CPU 的 node 中转。

**距离是相对值，不是 ns**。本机的换算大致是：`10→0 跳`、`20→1 跳`、`30→2 跳`、
`40→3 跳`，每跳约 +50~130ns。`61` 以上是另一套（封装内内存域），实测折算
（§6.5）：`61→141-145ns`、`71→187-205ns`、`81→268-285ns`、`91→360-477ns`。
注意 SLIT 的数值和延迟**不是线性的**：距离 61→91 只涨 1.5 倍（数字），延迟却涨
2.5-3 倍。

### 6.3 最容易踩的坑：绑了核，没绑内存

`numactl --physcpubind=190-227` 只限制"用哪些核"，**不限制"内存在哪"**。
内存落在哪由 **first touch** 决定：

- 单线程初始化一个 8GB 数组 → 全落在那个线程的 node 上。
- 之后 38 个线程一起读 → 37 个线程在跨 node 读，延迟从 130 变 180-255ns。
- **同一个程序、同一份代码，能差 1.5-2 倍。**

正确姿势：

```bash
# 核和内存都绑（本地口径）
numactl --physcpubind=190-227 --membind=5 ./your_program

# 或者让每个线程 first-touch 自己那段（更接近真实 HPC）
#   → 分配后立刻并行写一遍，再开始算
```

### 6.4 为什么"跨 socket 只有 17 GB/s"

本机的 socket 之间靠片间互连（HCCS 一类）。它主要服务**缓存一致性**（谁改了哪条
cacheline），不是为"一个 node 猛拉远端内存"设计的。实测：

- 延迟 354-387ns（本地的 2.7-3 倍）
- 带宽 17 GB/s（本地的 1/10）

**工程结论**：跨 socket 的正确用法是"每个 socket 处理自己的数据"（MPI 一个 rank
一个 socket），而不是"一边的核去读另一边的内存"。真要跨，就把数据复制过来，
不要反复远程读。

### 6.5 那 16 个 4GB 的 node 是"封装内内存"（CPU1/CPU2 MemoryOnChip）

`numactl -H` 会看到 node16-31：**没有 CPU、各 4096MB**。这一节原来我只能写
"身份待确认"，现在 `dmidecode` + `dmesg` 给了确定答案。

**它是 CPU 封装内的内存，不是插在主板上的内存条。** `dmidecode -t 17` 里有两条
与众不同的记录：

```
Handle 0x0014, DMI type 17, 92 bytes
Memory Device
        Size: 32 GB
        Form Factor: Die                 ← 不是 DIMM，是"裸 die"
        Locator: CPU1MemoryOnChip        ← 字面意思：CPU1 片上内存
        Bank Locator: Unknown
        Type: DRAM
        Speed: 10000 MT/s                ← 比 DDR5-5600 快得多
        Total Width: 64 bits / Data Width: 64 bits
        Array Handle: 0xFFFE             ← 不属于"主板内存阵列"
        Logical Size: 32 GB
```

另一条是 `Locator: CPU2MemoryOnChip`，同样 32 GB / 10000 MT/s。**两颗 CPU 各带
32GB 封装内内存，合计 64GB**——正好等于 node16-31 的 16 × 4GB。

**它怎么变成 16 个 NUMA node 的？** 看 `dmesg` 里的 SRAT 解析：

```
ACPI: SRAT: Node 16 PXM 16 [mem 0x1000000000-0x10ffffffff] hotplug
...
ACPI: SRAT: Node 23 PXM 23 [mem 0x1700000000-0x17ffffffff] hotplug
ACPI: SRAT: Node 24 PXM 24 [mem 0x81000000000-0x810ffffffff] hotplug
...
ACPI: SRAT: Node 31 PXM 31 [mem 0x81700000000-0x817ffffffff] hotplug
NUMA: Initmem setup node 16 [<memory-less node>]
NUMA: NODE_DATA(16) on node 15
```

把 SRAT 里 32 段地址整理出来，拓扑就一目了然了（这段在报告 §5 里能直接看到）：

| 范围 | 物理地址 | 每段大小 | 是什么 |
|---|---|---|---|
| node 0-7 | 1.00 → 1.50 TiB（node0 另有 0-2GiB 的低位窗口） | 64 GiB | **socket 0 的 8 条 DDR5** |
| node 8-15 | 9.00 → 9.50 TiB | 64 GiB | **socket 1 的 8 条 DDR5** |
| node 16-23 | 64 GiB → 96 GiB | **4 GiB** | **CPU1 封装内内存**（32GiB 切 8 段） |
| node 24-31 | 8.06 → 8.09 TiB | **4 GiB** | **CPU2 封装内内存**（32GiB 切 8 段） |

两个细节值得注意：

- **每颗 CPU 的封装内内存，地址都排在它自己的 DRAM 窗口下面**：CPU1 是 64GiB vs
  1.0TiB，CPU2 是 8.06TiB vs 9.0TiB。位置关系不是随机的。
- 标记是 **`hotplug`**：固件把它当"可热插拔内存"报出来，所以内核启动时先按
  "memory-less node"（无内存 node）建结构，之后才把内存挂上去——这就是
  `dmesg` 里 `Initmem setup node 16 [<memory-less node>]` 的来源。

**实测性能**（2026-11-04 下午两轮：延迟 16:15、带宽 19:00 修正轮；32 线程/窗口）：

先纠两条旧结论。上一版这里写的是"node17 6.1/47.0/64.4 GB/s、延迟 241ns"——那是
**从一个固定核、用 4MB 足迹**量出来的单点。4MB 足迹里有 1/5 落在 L2，量出来的不是
纯远端延迟。带宽那条更绕：旧表"单窗口 129-364"是拿**不同 SCCL 的核**分别去压不同
窗口压出来的，差的是核到窗口的距离，不是窗口（下面有反证）。

**延迟取决于"哪颗核访问哪个窗口"**（`chase 65536`，单位 ns）：

| SLIT 距离 | 实测延迟 | 典型关系 |
|---|---|---|
| 61 | **141-145** | 本 SCCL 的"专属"窗口（node i ↔ node 16+i，1:1 配对） |
| 71 | **187-205** | 同 SCCL 的另外三个窗口 |
| 81 | **268-285** | 同 socket 另一个 SCCL 的窗口 |
| 91 | **360-477** | 另一个 socket 的窗口 |
| — | 124-132 | 参照：本地 DRAM |

最近的那个窗口只比本地 DRAM 慢 **10%**，最远的要 3.7 倍。旧稿的 241ns 落在
"61/71/81"三档之间，哪个都不代表——这就是单点的毛病。

**带宽取决于"核站在哪"，不取决于窗口**（32 线程/窗口，1GB 缓冲区，2s 窗口）：

| 谁去访问 | SLIT 距离 | 单窗口带宽 |
|---|---|---|
| 配对的 CPU node（`node i` ↔ `node 16+i`） | 61 | **355-411 GB/s** |
| 同 SCCL 的另外三个 CPU node | 71 | **183-195 GB/s** |
| 同 socket 另一个 SCCL | 81 | **137-139 GB/s** |
| 另一个 socket | 91 | **18 GB/s** |
| 对照：本地 DRAM node（同一探针） | — | 167 GB/s |

跨 socket 的 18 GB/s 正好是 §6.1 里跨 socket DRAM 那个 17 GB/s——**是同一面墙**，
限制在 socket 之间的链路上，不是封装内内存本身。

**窗口之间没有共享链路，聚合是线性的**（每个窗口用自己的配对 32 核）：

| 同时压几个窗口 | 实测 |
|---|---|
| 1 | 382 |
| 2 | 773 |
| 4 | 1544 |
| socket0 的七个 | **2719** |
| socket1 的七个 | **2614** |
| 两个 socket 十四个 | **5238 GB/s** |

换个说法：**同样 7×32 线程、每进程 512MB**，摊到七个窗口是 **2621 GB/s**，全压在
一个窗口上只有 **284 GB/s**——摊开是它的 **9.2 倍**。

**多路能不能叠加：只有两条硬墙**（每路 32 核、1GB）：

| 路径 | 单路 | 多路同时 | 能不能叠 |
|---|---|---|---|
| 配对（61） | 378-410 | 3 窗口 1050-1126、4 窗口 1535、14 窗口 5238 | 完全线性 |
| 同 SCCL 非配对（71） | 193-200 | 3 路 452-492、4 路 613 | 部分叠加 |
| 跨 SCCL（81） | 130-138 | 3-4 路 **173** | **硬墙** |
| 跨 socket（91） | 18 | 3 路 **19** | **硬墙** |

跨 SCCL 的 173 与是哪一对 SCCL 无关（0→1、1→0、8→9、9→8 都量到 173-174），而且是
**按方向各算的**：两个方向同时跑，132 + 137 两路合计 250，近似全双工。所以"跨 SCCL
慢"不是一个窗口的端口不够，而是**SCCL 之间那条链路**就这么宽。

具体到带宽：旧表把每个窗口交给"socket 里随手切的一段 32 核"去压，而**没配对的核
只能拿到 137-195 GB/s**。旧表自己的距离一栏就能对上：192 GB/s 的是距离 81、
364 GB/s 的是距离 61。所以那 2.8 倍不是窗口的差异，是**核站在哪**的差异。

**反证：这些字节不走 DDR5。** 每个 SCCL 有 8 个 DDRC 计数器（`hisi_scclN_ddrcM_K`，
事件 `flux_rd`）。先拿本地 DRAM 校准：8 个 DDRC 各占 12.5%，按 32 字节/事务折算
正好等于探针量到的 164.5 GB/s。再压 node30（389 GB/s）：32 个 DDRC 的读事务合计只
折出 0.59 GB/s——**封装内内存有自己的接口，完全不经过主板内存通道**。

**一个窗口 = 一个 UC**（unified cache / 内存控制器单元）。压 node30 时，只有
`hisi_sccl9_uc2`（node14 的 UC）大幅计数，另外 15 个只有背景。每 SCCL 4 个 UC、
每 socket 16 个、全机 32 个，与"每 socket 16 个内存 node"严格 1:1。顺带两个可以
直接用的东西：

- `cring_txdat_cnt` 是这层的字节计数器，**1 个计数 = 256 B**（409.6 GB/s × 2 s /
  3.2e9 = 256，8 线程那组也对得上），不跑探针也能量带宽。
- UC 的四个子设备 `ucN_0..3` 读数**完全相同**，是同一单元的多路读口、不是四条通道。
  所以"窗口内部几条通道"从软件侧读不出来；能确定的只是 32MB 的切片单独压就有整窗口
  六到八成的带宽（交织很细，不是"每 N GB 一条通道"的粗分块）。

**固件里的 `10000 MT/s` 不能换算成"每颗 CPU 80 GB/s"**。上一版按 10000 MT/s ×
8 字节算出 80 GB/s，再用"单点 64.4 GB/s 达到 80%"去解释；两个数都错了。实测单窗口
355-411，一个 socket 的七个窗口 2.6 TB/s。`dmidecode` 的 `Speed: 10000 MT/s` 只是
DIMM 字段，没有给位宽，不能当接口带宽用。

另外两条实测事实：

- **32 条核就到顶**：16 线程 235、24 线程 285、**32 线程 423**、38 线程 394。
  再往上"掉"不是内存效率问题——48 / 76 线程压在 34-38 条核上，纯粹是抢核。
  上一版写的"48/64 线程把 row buffer 冲散"是错的。
- **窗口之间很均匀**：十四条里最慢 346、最快 411，差 1.2 倍。剩下这点差异没有单独
  追（可能就是共享机噪声）。

**固件自己怎么说（HMAT，本次解析）**：HMAT 里有两张等级表，都是 16×32 的矩阵。
"访问延迟"那张的值是 10/20/30/40，封装内窗口是 31/41/51/61——**就是 SLIT 距离照抄
一遍，只把封装内那 16 列统一减了 30**；基本单位写的是 1000 ps，可 10 × 1ns = 10ns
显然不是物理延迟（实测 130ns）。另一张表的数据类型是 3（规范只定义 0/1/2），值
25 对 250，即"封装内 = DRAM 的 10 倍带宽"。**倍数这一项碰巧差不多**（实测总量
5238 vs 整机 DRAM 678，7.7 倍），**但绝对值对不上**（25 × 1024 MB/s = 25.6 GB/s，
实测单窗口 380）。所以 HMAT 只能当"等级/倍数"看，不能当 ns / GB/s 用。内存侧缓存
那张表则全是 0，等于没声明。

> 解析脚本：`tools/parse-acpi-hw.py`，原始表由 `tools/collect-hw-info.sh` 以 root
> 导出。顺带记两个坑：本机 HMAT / SRAT 在 36 字节表头之后还各塞了 **4 / 12 字节
> 填充**，HMAT 的等级条目是 **16 位**而不是 32 位——按教科书布局解会读出全 0。

**要不要用？**

- 正常程序不用担心：它没有 CPU，普通 first-touch **永远落不到**上面；只有显式
  `--membind=16` 或 `move_pages` 才会用到。
- 对延迟敏感的负载**别往上面放**：最近也要 141ns（本地 130 的 1.1 倍），远一档就
  268-285ns（2.1 倍）。但如果核和窗口配对了（node i 用 node 16+i），代价只有 10%。
- 对"能放下就赚到"的大缓存（比如只读的查表数据、临时缓冲区）可以考虑——**配对之后
  它是本机最快的"内存"**：单窗口 380-410 GB/s 是 DRAM node（167）的 2.3 倍，十四个
  窗口合计 5.2 TB/s 是整机 DRAM（678）的 7.7 倍。带宽上配对更要紧：跨 SCCL 只剩
  137-195，跨 socket 只剩 18 GB/s。注意 node16 与 node31 已被内核 slab 占满（各只剩
  6-10MB），共享机器上随时可能被别人拿走。
- 如果你在 `numastat` 里看到匿名页落在 node16-31，而你没主动绑过，那一定是绑错了
  或者 `numactl --interleave=all` 之类的策略把它算了进去。

---

## 7. 回到性能优化：这些数怎么用

### 7.1 先判断你的负载是"带宽型"还是"延迟型"

| 特征 | 带宽型 | 延迟型（`x[idx]` 随机取数） |
|---|---|---|
| 访问模式 | 顺序/大块 | 随机/小粒度 |
| 加线程 | 有用，到 16 核/通道组饱和 | 有用，但边际递减，且会互相踩 row |
| 单核上限 | 19 GB/s | 6.8 GB/s（随机 64B） |
| 优化方向 | 少搬字节、用满通道 | **提高同时在飞的请求数**、预取、改访问顺序 |

`x[idx]` 这类随机行取数属于后者：单核 6.8 GB/s，离 19 GB/s 的顺序上限差 2.8 倍，离 170 GB/s
的通道上限差 25 倍。**所以它的瓶颈从来不是"内存不够快"，而是"一次只等一条"。**

### 7.2 线程数别拍脑袋

- 内存密集 → **每 SCCL 16-24 线程就够**，再多是浪费（还可能因为争抢变慢）。
- 本机一个 node 38 核，但 16 核就到 164.7/168.3 = 98%。
- 跨 node 之前先问：**这两个 node 是不是同一个 SCCL？** 是的话，加 node 不加带宽。

### 7.3 跨 NUMA 会掩盖一切优化

如果生产上源数据落在远端 node，内核优化再好也追不回来：

- 同 socket 另一 SCCL：延迟 255ns（本地 2 倍）
- 跨 socket：延迟 355ns + 带宽 17 GB/s

**先查绑核/绑内存策略，再谈优化。** 这是部署问题，不是代码问题。

---

## 8. 命令速查：想查什么，跑哪条

### 8.1 不需要 root（现在就都能跑）

| 想知道 | 命令 |
|---|---|
| 拓扑、NUMA node、距离 | `numactl -H` |
| 层次结构（谁包含谁） | `lstopo-no-graphics --of console` |
| 每核属于哪个 node/socket | `lscpu -e` |
| 缓存大小与相联度 | `lscpu -C` |
| 每个 node 的内存用量 | `cat /sys/devices/system/node/node*/meminfo` |
| 进程的内存落在哪些 node | `numastat -p <pid>` |
| **内存通道清单** | `ls /sys/bus/event_source/devices/ \| grep ddrc` |
| **每通道读/写/周期** | `perf stat -a -e hisi_sccl1_ddrc0_0/flux_rd/,hisi_sccl1_ddrc0_0/cycles/ sleep 1` |
| 单点延迟 | `memprobe chase <KB> <次数>`（见附录） |
| 带宽 | `memprobe bandwidth <MB> <线程> rd\|cp <重复>` |

一次看全部 32 个子通道（= 16 条通道）的读命令：

```bash
EV=$(for s in 0 1 8 9; do for d in 0 1 2 3; do for c in 0 1; do
  echo -n "hisi_sccl${s}_ddrc${d}_${c}/flux_rd/,"; done; done; done | sed 's/,$//')
perf stat -a -e "$EV" <你的程序>
```

### 8.2 需要 root（你方便时跑一下，我来解读）

**一条命令全采完**（推荐，输出一个文本文件直接给我）：

```bash
sudo bash tools/collect-hw-info.sh            # 输出到当前目录
sudo bash tools/collect-hw-info.sh --redact   # 顺便去掉序列号/UUID
```

它会把下面这些全抓进 `hw-report-<时间>.txt`，并把 ACPI 原始表另存到
`hw-raw-<时间>/`：

| 命令 | 能回答什么 |
|---|---|
| `sudo dmidecode -t memory` | **DIMM 数量、单条容量、频率、厂家、插槽位置（Locator/Bank Locator）**——通道数的最权威来源 |
| `sudo dmidecode -t 16` | 物理内存阵列：最大容量、插槽数 |
| `sudo dmidecode -t 17` | 同上，逐条内存的详细信息（本文的 DDR5-5600、80 位宽、`CPU1MemoryOnChip` 都出自这里） |
| `sudo dmidecode -t cache` | 缓存层级（与 `lscpu -C` 对照） |
| `sudo dmesg \| grep -iE "numa\|srat\|hmat"` | 内核怎么解析 NUMA/内存拓扑（**本文 node16-31 的答案就来自这里**） |
| `sudo cat /sys/firmware/acpi/tables/SRAT` | NUMA 拓扑的固件原文（每段内存的物理地址区间） |
| `sudo cat /sys/firmware/acpi/tables/HMAT` | 内存侧缓存/带宽的固件描述（本机存在，3848 字节） |
| `sudo cat /proc/iomem` | **真实物理地址**（非 root 会被清零） |
| `sudo numastat -m` | 每个 node 的内存分类（本机这个版本会报 "Token ... not in hash table"，是 numactl 与内核字段对不上，不影响别的） |

> 这张表原本是"我需要你跑一下、我来解读"的清单。**现在已经跑过了**，结果写在
> §2.2（DIMM 清单）和 §6.5（node16-31 的身份）。留在这里是为了让你知道
> "同一个问题该问哪个工具"，以及换一台机器时怎么复现。

---

## 9. 常见误解（本机实测会打脸的版本）

| 误解 | 实测 |
|---|---|
| "内存带宽是所有核共享的一大块" | 是 **16 条独立通道**，按 SCCL 分成 4 组（每组 4 条），跨组才叠加 |
| "PMU 里有 32 个 ddrc，所以是 32 条通道" | 只有 **16 条**；DDR5 每条再分 2 个 32 位子通道（§2.3） |
| "核越多带宽越高" | 单 node 16 核就到 98%；38 核不加带宽 |
| "NUMA node 越多带宽越大" | node5 和 node6 是同一组通道，加 node 不加带宽 |
| "顺序和随机差不多，反正都在内存里" | 单核差 2.8x，多核差 4x，随机还随线程变差 |
| "写和读一样快" | 写要 RFO；38 线程拷贝比 16 线程还慢 |
| "绑了核就绑了内存" | 内存位置由 first touch 决定，必须 `--membind` |
| "单核能跑满带宽" | 单核 19 GB/s，只有整机的 3%，连一条通道的 42% 都不到 |
| "有 L3 兜底" | **本机没有 L3**，L2 之后直接 128ns |
| "多出来的 16 个 4GB node 是坏内存/预留区" | 是 **CPU 封装内的 32GB×2 内存**，10000 MT/s，被切成 16 个 4GB node（§6.5） |

---

## 10. 这份实测的局限（诚实边界）

1. **机器是共享的**：测的时候有其他租户，同一命令重复跑会有 ±10% 波动；
   跨 SCCL 的聚合数我用"同一时间窗口"的方法才稳定（附录 A）。
2. **封装内内存：链路墙查清了，通道数没有**。单窗口 355-411 GB/s、十四个窗口合计
   5238 GB/s（§6.5），"8 个 node 共享一条 80 GB/s"完全不成立，DDRC 计数器也证实
   这些字节不走 DDR5。跨 SCCL 的 173 GB/s 已证实是**链路带宽**（与哪一对 SCCL 无关、
   双向合计 250），跨 socket 的 19 GB/s 是另一条链路。**没解释的是**：为什么恰好
   173 GB/s；同 SCCL 内非配对为什么先砍一半（193-200，但多路能叠到 450-900）；
   以及一个 UC 内部到底几条通道（PMU 看不到）。
3. **没测写带宽的独立上限**：`cp` 是读+写合计，分不出纯写。
4. **没测多 socket 聚合的交叉矩阵**：只测了"一个 node 读远端"，没测"16 个 node
   同时读远端"。
5. **没测 SMT/频率变化的影响**：本机无 SMT，且 boost 关闭（`CPU max MHz: 2000`、
   实测固定 2.0GHz），所以本文数字不涉及频率漂移。
6. **BIOS 的型号名有点可疑**：`lscpu` 报 `Kunpeng 920 72F8`，但同时是 2×304 核 +
   16 条 DDR5 通道 + 32 个 NUMA node——这不是常见 Kunpeng 920 的形态。以
   `dmidecode` / SRAT 的实际结构为准，型号名只当参考。
7. **HMAT 已经解析出来了，但它的绝对值不能当物理量用**（§6.5 末）：延迟表是 SLIT
   距离照抄（封装内那 16 列统一减 30），带宽表的数据类型是规范外的 3、值 25:250。
   **倍数**（10 倍）碰巧和实测的 7.7 倍差不多，但**基本单位折出来的 25.6 GB/s /
   256 GB/s** 与实测（380 / 167）对不上。只能当"等级"看。表里还有一张"内存侧缓存"
   子表，全 0，等于没声明。
8. **SRAT/HMAT 的表头后面有填充、条目是 16 位**：HMAT 在 36 字节表头后多 4 字节、
   SRAT 多 12 字节，HMAT 的等级条目是 `u16` 不是 `u32`；按教科书布局解会读出全 0。
   解析器已按"从哪个偏移能正好走完整张表"自动定位（`tools/parse-acpi-hw.py`），
   但**为什么填这些字节，不知道**。

---

## 附录 A：时间窗口聚合带宽探针（`memwin.c`）

**为什么需要它**：`memprobe bandwidth` 是单线程 first-touch，`--membind` 给多个
node 时所有页会落在同一个 node 上；而"每个 node 起一个进程各测各的"又可能因为
读的窗口（十几毫秒）不重叠而虚高（我一开始就量出 1246 GB/s，实际是 658）。

**解法**：所有进程用**同一个绝对时钟起点**，在固定时长内反复读，输出读到的字节数；
聚合带宽 = Σ字节 ÷ 时长。这样重叠是确定的。

```c
// gcc -O3 -fopenmp -o memwin memwin.c
// 用法: ./memwin BUF_MB SEC START_EPOCH [THREADS]
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <omp.h>

static double re_now(void) { struct timespec t; clock_gettime(CLOCK_REALTIME, &t);
    return t.tv_sec + 1e-9 * t.tv_nsec; }
static double mono_now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + 1e-9 * t.tv_nsec; }

int main(int argc, char **argv) {
    size_t size = (size_t)atol(argv[1]) * 1024ULL * 1024ULL;
    double sec = atof(argv[2]), start = atof(argv[3]);
    int nt = argc > 4 ? atoi(argv[4]) : 1;
    size_t n = size / 8;
    uint64_t *a = aligned_alloc(64, size);

    // 并行 first touch：每个线程摸自己那段，页才会落在本 node
    #pragma omp parallel num_threads(nt)
    {
        size_t lo = (size_t)((double)n * omp_get_thread_num() / nt);
        size_t hi = (size_t)((double)n * (omp_get_thread_num() + 1) / nt);
        for (size_t i = lo; i < hi; i++) a[i] = i * 3 + 1;
    }

    while (re_now() < start) { }          // 所有进程对齐到同一时刻
    double t0 = mono_now();
    unsigned long long bytes = 0;
    while (mono_now() - t0 < sec) {
        #pragma omp parallel num_threads(nt) reduction(+:bytes)
        {
            size_t lo = (size_t)((double)n * omp_get_thread_num() / nt);
            size_t hi = (size_t)((double)n * (omp_get_thread_num() + 1) / nt);
            uint64_t s0 = 0, s1 = 0, s2 = 0, s3 = 0;
            for (size_t i = lo; i + 3 < hi; i += 4) {
                s0 += a[i]; s1 += a[i+1]; s2 += a[i+2]; s3 += a[i+3];
            }
            bytes += (hi - lo) * 8ULL;
            if (omp_get_thread_num() == 0) a[0] += s0 + s1 + s2 + s3;
        }
    }
    printf("%.0f\n", (double)bytes);
    return 0;
}
```

用法（16 个 node 一起读，2 秒）：

```bash
start=$(python -c "import time;print(time.time()+2.5)")
for n in $(seq 0 15); do
  lo=$((n*38)); hi=$((lo+37))
  numactl --physcpubind=$lo-$hi --membind=$n ./memwin 1024 2 $start 38 > w_$n.txt &
done
wait
python -c "print(sum(int(open(f'w_{n}.txt').read()) for n in range(16))/1e9/2, 'GB/s')"
# → 658 GB/s
```

---

## 附录 B：这次实测的完整命令清单（按调查顺序）

下面就是本文所有数字的出处，按"我是怎么一步步查下去"的顺序排。你可以照着复现。

### B.1 先看拓扑：这台机器有几个"家"

```bash
lscpu | grep -E "Model name|CPU\(s\)|Thread|Core|Socket|NUMA|MHz"
#   → 608 核、1 thread/core、2 socket、每 socket 304 核、32 NUMA node

lscpu -e | head -40
#   → 每核的 NODE / SOCKET / CORE / L1d:L1i:L2 映射

lscpu -C
#   → L1d 32K 8路、L1i 32K 4路、L2 768K 12路；没有 L3

numactl -H
#   → 32 个 node、每个的大小、node0-15 有 CPU、node16-31 没有

for n in /sys/devices/system/node/node*; do
  echo -n "$(basename $n): "; cat $n/cpulist; echo
done
#   → node5 = 190-227（38 核），确认"每 node 38 核"

cat /sys/devices/system/node/node5/distance
#   → 30 30 30 30 20 10 20 20 40 ...（本地/同 SCCL/同 socket/跨 socket）
```

### B.2 关键一步：从计数器推出通道数（然后被 dmidecode 纠正）

```bash
ls /sys/bus/event_source/devices/ | grep ddrc | wc -l
#   → 32

ls /sys/bus/event_source/devices/ | grep ddrc
#   → hisi_sccl{0,1,8,9}_ddrc{0,1,2,3}_{0,1}
#     4 个 SCCL × 4 个控制器 × 2 个子通道 = 32 个实例
```

看一个实例暴露了什么：

```bash
ls /sys/bus/event_source/devices/hisi_sccl0_ddrc0_0/events/
#   → cycles  flux_rd  flux_wr
cat /sys/bus/event_source/devices/hisi_sccl0_ddrc0_0/events/flux_rd
#   → config=0x84
```

**这里最容易错**：看到 32 就以为 32 条通道。用 `dmidecode` 一对就露馅了——只有
16 条 DIMM，插在 `CHANNEL 0..7`（每 socket 8 条）。DDR5 的一条通道内部是两个
32 位子通道，所以实例数才是 2 倍。

```bash
sudo dmidecode -t 17 | grep -E "Locator:|Bank Locator:|Size:|Type:|Speed:|Rank|Width"
#   → 16 条 64GB DDR5-5600 2R，SOCKET 0/1 CHANNEL 0..7 DIMM 0
#   → 另有 2 条 "Die/CPU1MemoryOnChip"，32GB/10000 MT/s（封装内内存）
```

### B.3 测内存频率（不用 root）

```bash
perf stat -a -e hisi_sccl1_ddrc0_0/cycles/ sleep 1
#   → 1,405,xxx,xxx cycles / 1s = 1405 MHz
#     DDR5 控制器时钟 = 数据率 / 4 → 1405 × 4 ≈ 5600 MT/s
#     与 dmidecode 的 "Speed: 5600 MT/s" 完全对上
```

### B.4 确认"哪个 NUMA node 用哪 4 条通道"

思路：绑核 + 绑内存到同一个 node，读 1GB，看哪个 SCCL 的计数器在涨。

```bash
EV=$(for s in 0 1 8 9; do for d in 0 1 2 3; do for c in 0 1; do
  echo -n "hisi_sccl${s}_ddrc${d}_${c}/flux_rd/,"; done; done; done | sed 's/,$//')

perf stat -a -e "$EV" numactl --physcpubind=200 --membind=5 \
  ./memprobe bandwidth 1024 1 rd 1
#   → sccl1 的 8 个子通道各 ~4.4M，其余是背景
#   → node5 属于 sccl1（node4-7 同组）
```

对 node0/1/4/6/7/8/12 重复，得到映射表（§2.4）。

**别忘了量背景**：

```bash
perf stat -a -e "$EV" sleep 1
#   → sccl9 空载就有 ~21M/s（其他租户），所以"有数"不等于"是你在用"
```

### B.5 测延迟

```bash
# 编译探针（源码在 benchmarks/memprobe.c）
gcc -O3 -fopenmp -o memprobe benchmarks/memprobe.c

# 单点延迟：工作集 64MB（DRAM）
numactl --physcpubind=200 --membind=5 ./memprobe chase 65536 8000000
#   → ns/access=129.9

# 换内存位置，量 NUMA 阶梯
for M in 5 6 0 8; do
  numactl --physcpubind=200 --membind=$M ./memprobe chase 65536 8000000
done
#   → 129.9 / 180.6 / 254.9 / 354.1 ns
```

### B.6 测带宽（单 node / 单 SCCL / 整机）

```bash
# 线程扩展（单 node）
for T in 1 2 4 8 16 24 38; do
  numactl --physcpubind=190-227 --membind=5 \
    ./memprobe bandwidth 1024 $T rd 2
done
#   → 19.0 / 37.8 / 73.0 / 128.6 / 164.7 / 169.4 / 168.3 GB/s

# 拷贝
numactl --physcpubind=190-227 --membind=5 ./memprobe bandwidth 1024 16 cp 2
#   → 145.1 GB/s（读+写合计）
```

**整机聚合不能这么测**（单线程 first-touch + 窗口不重叠会虚高），要用附录 A 的
`memwin`：

```bash
start=$(python -c "import time;print(time.time()+2.5)")
for n in $(seq 0 15); do
  lo=$((n*38)); hi=$((lo+37))
  numactl --physcpubind=$lo-$hi --membind=$n ./memwin 1024 2 $start 38 > w_$n.txt &
done
wait
python -c "print(sum(int(open(f'w_{n}.txt').read()) for n in range(16))/1e9/2)"
#   → 658 GB/s
```

### B.7 查"那 16 个 4GB 的 node"（含 root 部分）

```bash
numactl -H | grep -E "^node (1[6-9]|2[0-9]|3[01]) "
#   → 每个 4096 MB，cpus 为空

grep -E "MemTotal|MemFree|AnonPages" /sys/devices/system/node/node16/meminfo
#   → 4GB 里只剩 11MB，AnonPages=0

grep -E "nr_free_pages|nr_slab_unreclaimable" /sys/devices/system/node/node16/vmstat
#   → 723MB 不可回收 slab —— 有驱动在用

# 注意：4MB 足迹有一半在 L2，量出来不是纯远端延迟；用 64MB
numactl --physcpubind=38 --membind=17 ./memprobe chase 65536 2000000
#   → 141-145 ns（node1 的核 → node17，SLIT 距离 61，1:1 配对）
numactl --physcpubind=0-31 --membind=17 ./memprobe chase 65536 2000000
#   → 268-285 ns（node0 的核 → node17，距离 81，同 socket 另一 SCCL）
numactl --physcpubind=38-69 --membind=17 ./memprobe bandwidth 1024 32 rd 2
#   → 380-410 GB/s（配对核 + 1GB 缓冲；跨 SCCL 的核只有 137-195，跨 socket 18）

# 整套带宽结构（距离阶梯、14 窗口扫描、聚合、集中 vs 摊开、DDRC 反证）：
python3 tools/probe-onpkg-bw.py

# 身份：这两条是答案（root）
sudo dmidecode -t 17 | grep -B6 -A14 "MemoryOnChip"
#   → Form Factor: Die / Locator: CPU1MemoryOnChip / 32 GB / 10000 MT/s
#     Array Handle: 0xFFFE（独立于主板内存阵列）

sudo dmesg | grep -E "SRAT: Node (1[6-9]|2[0-9]|3[01]) "
#   → 每段 4GB，标 hotplug：0x1000000000-0x10ffffffff ... 0x81700000000-0x817ffffffff
sudo dmesg | grep "memory-less node"
#   → NUMA: Initmem setup node 16 [<memory-less node>]
```

### B.8 顺序 vs 随机

```bash
numactl --physcpubind=190-227 --membind=5 \
  ./memprobe gathermt 1024 5000000 38 rand
#   → ns/access=57.8, line_GBps=42.1

numactl --physcpubind=190-227 --membind=5 \
  ./memprobe bandwidth 1024 38 rd 2
#   → 168.3 GB/s
```

### B.9 内存使用现状

```bash
numastat -m                      # 每 node 的内存分类（非 root 可能报 Token 错）
awk '/^Node [0-9]+, zone/{...}' /proc/zoneinfo   # 每 node managed pages
cat /sys/devices/system/node/node*/meminfo
cat /proc/meminfo | grep -E "Huge|MemTotal|MemFree"
cat /sys/kernel/mm/transparent_hugepage/enabled
```

### B.10 root 部分：一次性采集

```bash
sudo bash tools/collect-hw-info.sh
#   → 一次性采集 dmidecode / SRAT / HMAT / iomem / numastat / PMU 空载读数
#   → 输出 hw-report-<时间>.txt + hw-raw-<时间>/
```

本文用的就是 `hw-report-20261104-144311.txt`（2277 行，脱敏后放在
`docs/hw-report-20261104-144311.txt`，序列号/资产标签已抹掉）。脚本只读、5 秒跑完、
支持 `--redact` 去序列号。跑完把路径发给解读的人即可。

> 想自己看全文：`docs/hw-report-20261104-144311.txt` 里 §3 是 `dmidecode` 的原始
> 输出（16 条 DIMM 逐条字段），§4 是 SRAT/SLIT/HMAT 的字符串视图，§5 是 `dmesg`
> 里内核怎么解析拓扑，§6 是 `/proc/iomem` 的真实物理地址。**看 §3 和 §5 就够
> 回答本文 90% 的问题。**

---

**数据轮次**：2026-11-04 下午，node 5 为主，`memprobe` / `memwin` 本地编译，
`perf` 读 `hisi_sccl*_ddrc*` 计数器。机器为部分恢复期，带宽与延迟已与快态轮次
交叉校验（差 2-4%）。
