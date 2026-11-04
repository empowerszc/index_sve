#!/usr/bin/env python3
"""解析 HMAT / SLIT / SRAT，输出内存拓扑与固件声明的延迟/带宽等级。

用法: python3 tools/parse-acpi-hw.py <目录>（目录里放 HMAT[.bin]、SLIT[.bin]、SRAT[.bin]）

本机固件（HISI HIP11）的三张表都不按教科书走：HMAT 与 SRAT 在 36 字节表头之后还有
若干字节填充（HMAT 4 字节、SRAT 12 字节），HMAT 的等级条目是 16 位而不是 32 位。
脚本按"从哪个偏移开始能正好走完整张表"自动定位子表起点，并把填充字节数打出来。
"""
import os
import struct
import sys

HMAT_TYPES = {0: "地址区间", 1: "延迟/带宽等级", 2: "内存侧缓存"}
DTYPES = {0: "访问延迟", 1: "读带宽", 2: "写带宽"}
SRAT_TYPES = {0: "处理器 LAPIC", 1: "内存亲和", 2: "中断", 3: "GICC", 4: "GIC ITS", 5: "通用发起者"}


def header(data):
    sig, length, rev = struct.unpack_from("<4sIB", data, 0)
    oem = data[10:16].decode(errors="replace").strip("\x00 ")
    tid = data[16:24].decode(errors="replace").strip("\x00 ")
    return {"sig": sig.decode(errors="replace"), "length": length, "rev": rev, "oem": oem, "table_id": tid}


def find_body(data, kind, search=range(36, 73)):
    """返回 (子表起点, 子表个数)；找不到返回 (None, 0)。"""
    total = header(data)["length"]
    for start in search:
        off, n = start, 0
        while off + 2 <= total:
            if kind == "hmat":
                if off + 8 > total:
                    break
                _t, _r, ln = struct.unpack_from("<HHI", data, off)
                if ln < 8:
                    break
            else:
                ln = data[off + 1]
                if ln < 2:
                    break
            if off + ln > total:
                break
            off += ln
            n += 1
        if off == total and n:
            return start, n
    return None, 0


def hx(b, n=64):
    return " ".join(f"{x:02x}" for x in b[:n])


def parse_hmat(data):
    h = header(data)
    start, n = find_body(data, "hmat")
    print("=" * 78)
    print(f"[HMAT] 版本={h['rev']} 长度={h['length']} 文件={len(data)} OEM={h['oem']}/{h['table_id']}")
    if start is None:
        print("  ! 子表起点没找到（不是已知布局），整表前 64 字节：" + hx(data))
        return [], []
    print(f"  子表起点 = {start}（表头之后填充 {start - 36} 字节），共 {n} 条")
    subs = []
    off = start
    while off + 8 <= h["length"]:
        typ, _resv, ln = struct.unpack_from("<HHI", data, off)
        subs.append((typ, off, ln, data[off : off + ln]))
        if ln < 8:
            break
        off += ln

    counts = {}
    for typ, *_ in subs:
        counts[typ] = counts.get(typ, 0) + 1
    print("  子表：" + "、".join(f"{HMAT_TYPES.get(t, t)}×{c}" for t, c in sorted(counts.items())))

    ranges, locality = [], []
    for typ, off, ln, b in subs:
        if typ == 0:
            flags = struct.unpack_from("<H", b, 8)[0]
            ppd = struct.unpack_from("<I", b, 12)[0]
            mpd = struct.unpack_from("<I", b, 16)[0]
            if ln >= 48:
                base, mlen = struct.unpack_from("<QQ", b, 32)
            else:
                base = mlen = None
            ranges.append((ppd, mpd, base, mlen, flags, ln))
        elif typ == 1:
            locality.append((off, ln, b))
        elif typ == 2:
            pd = struct.unpack_from("<I", b, 8)[0]
            size = struct.unpack_from("<Q", b, 16)[0]
            attrs = struct.unpack_from("<I", b, 24)[0]
            print(f"  内存侧缓存：内存域={pd} 容量={size} 字节 属性=0x{attrs:08x}"
                  + ("（全 0 → 没有声明缓存）" if size == 0 and attrs == 0 else ""))
        else:
            print(f"  ! 未知子表类型 {typ}，长度 {ln}，原始：{hx(b)}")

    if ranges:
        with_addr = [r for r in ranges if r[2] is not None]
        print(f"\n  地址区间子表 {len(ranges)} 条，其中带地址的 {len(with_addr)} 条")
        if not with_addr:
            print("    → 本固件这些子表只有 40 字节，只声明「处理器域 ↔ 内存域」的对应关系，"
                  "不含 base/length；地址以 SRAT 为准")
            for ppd, mpd, _b, _l, flags, ln in ranges[:4]:
                print(f"    ppd={ppd:>2} mpd={mpd:>2} flags=0x{flags:02x}（子表长 {ln}）")
            print("    …（其余同构）")
        else:
            for ppd, mpd, base, mlen, flags, ln in with_addr:
                print(f"    ppd={ppd:>2} mpd={mpd:>2} 0x{base:012x} {mlen / 2**30:7.1f}GiB flags=0x{flags:02x}")

    for idx, (off, ln, b) in enumerate(locality, 1):
        flags, dtype = b[8], b[9]
        ni, nt = struct.unpack_from("<II", b, 12)
        base_unit = struct.unpack_from("<I", b, 24)[0]
        need = 32 + 4 * (ni + nt) + 2 * ni * nt
        name = DTYPES.get(dtype, f"数据类型{dtype}（规范外的值）")
        print(f"\n  [等级表 #{idx}] {name} 发起域={ni} 目标域={nt} 基本单位={base_unit} "
              f"子表长={ln}（布局算出 {need}）")
        if need != ln:
            print("    ! 长度对不上，只打原始字节：" + hx(b, 96))
            continue
        init = struct.unpack_from(f"<{ni}I", b, 32)
        tgt = struct.unpack_from(f"<{nt}I", b, 32 + 4 * ni)
        vals = struct.unpack_from(f"<{ni * nt}H", b, 32 + 4 * (ni + nt))
        scale = base_unit / 1000.0
        unit = "ns" if dtype == 0 else ("GB/s" if dtype in (1, 2) else "（按基本单位换算）")
        print(f"    发起域列表={list(init)}")
        print(f"    目标域列表={list(tgt)}")
        print(f"    原始值范围 {min(vals)}..{max(vals)}；按 条目×{base_unit}"
              f"{'ps' if dtype == 0 else 'MB/s'} 折算 = 最小 {min(vals) * scale:.1f} {unit}、"
              f"最大 {max(vals) * scale:.1f} {unit}")
        if dtype not in (0, 1, 2):
            print("    注意：这是规范外的数据类型，下面按原始值×基本单位折算，仅供参考")
        print("      发起\\目标 " + " ".join(f"{t:>5}" for t in tgt))
        for a in range(ni):
            row = vals[a * nt : (a + 1) * nt]
            print(f"      {init[a]:>8}  " + " ".join(f"{v:>5}" for v in row))
    return ranges, locality


def parse_slit(data):
    h = header(data)
    n = struct.unpack_from("<Q", data, 36)[0]
    mat = data[44 : 44 + n * n]
    print("\n" + "=" * 78)
    print(f"[SLIT] 版本={h['rev']} 长度={h['length']} 局部数={n}")
    if len(mat) != n * n:
        print(f"  ! 矩阵长度不符：需要 {n * n}，实际 {len(mat)}")
        return None
    m = [list(mat[i * n : (i + 1) * n]) for i in range(n)]
    print("  每个 CPU node 看封装内内存（node16-31）的距离分组：")
    for a in range(16):
        grp = {}
        for b in range(16, 32):
            grp.setdefault(m[a][b], []).append(b)
        print(f"    node{a:<2} → " + "；".join(f"{k}: {v}" for k, v in sorted(grp.items())))
    return m


def parse_srat(data):
    h = header(data)
    start, n = find_body(data, "srat")
    print("\n" + "=" * 78)
    print(f"[SRAT] 版本={h['rev']} 长度={h['length']} OEM={h['oem']}/{h['table_id']}")
    if start is None:
        print("  ! 子表起点没找到，整表前 64 字节：" + hx(data))
        return []
    print(f"  子表起点 = {start}（表头之后填充 {start - 36} 字节），共 {n} 条")
    hist, mem, gen = {}, [], []
    off = start
    while off + 2 <= h["length"]:
        typ, ln = data[off], data[off + 1]
        if ln < 2 or off + ln > h["length"]:
            print(f"  ! 在偏移 {off} 走不下去")
            break
        hist[typ] = hist.get(typ, 0) + 1
        if typ == 1:
            pd = struct.unpack_from("<I", data, off + 2)[0]
            base = struct.unpack_from("<Q", data, off + 8)[0]
            mlen = struct.unpack_from("<Q", data, off + 16)[0]
            flags = struct.unpack_from("<I", data, off + 28)[0]
            mem.append((pd, base, mlen, flags))
        elif typ == 5:
            pd = struct.unpack_from("<I", data, off + 4)[0]
            gen.append((pd, data[off + 8 : off + 24].rstrip(b"\x00")))
        off += ln
    print("  子表计数：" + "、".join(f"{SRAT_TYPES.get(t, t)}×{c}" for t, c in sorted(hist.items())))
    print("  内存亲和（flags bit0=启用 bit1=可热插拔）：")
    for pd, base, mlen, flags in mem:
        print(f"    pd={pd:>2} 0x{base:012x} {mlen / 2**30:7.1f}GiB flags=0x{flags:02x}")
    for pd, handle in gen:
        print(f"    通用发起者 pd={pd} handle={handle!r}")
    print("\n  封装内内存的地址段是否连续（每颗 CPU 8 × 4GiB）：")
    for group in (range(16, 24), range(24, 32)):
        rows = [m for m in mem if m[0] in group]
        gaps = {(rows[i + 1][1] - (rows[i][1] + rows[i][2])) for i in range(len(rows) - 1)}
        print(f"    node{group.start}-{group.stop - 1}: {len(rows)} 段，段间空隙 = {gaps or '只有一段'}")
    return mem


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "."
    print(f"# ACPI 表解析（目录 {d}；三个文件为 HMAT/SLIT/SRAT 或加 .bin 后缀）")
    paths = {}
    for n in ("HMAT", "SLIT", "SRAT"):
        for cand in (n, n + ".bin"):
            p = os.path.join(d, cand)
            if os.path.exists(p):
                paths[n] = p
                break
    missing = [n for n in ("HMAT", "SLIT", "SRAT") if n not in paths]
    if missing:
        print("缺少表：" + "、".join(missing))
        return 1

    hmat_ranges, _ = parse_hmat(open(paths["HMAT"], "rb").read())
    parse_slit(open(paths["SLIT"], "rb").read())
    srat_mem = parse_srat(open(paths["SRAT"], "rb").read())

    print("\n" + "=" * 78)
    print("[交叉校验]")
    if hmat_ranges and all(r[2] is not None for r in hmat_ranges):
        srat_set = {(base, mlen) for _pd, base, mlen, _f in srat_mem}
        ok = sum(1 for _p, _m, base, mlen, *_ in hmat_ranges if (base, mlen) in srat_set)
        print(f"  HMAT 地址区间与 SRAT 完全一致 {ok}/{len(hmat_ranges)}"
              + ("（布局解析正确）" if ok == len(hmat_ranges) else "（不一致，检查上面的偏移）"))
    else:
        print("  HMAT 不含地址区间，无法与 SRAT 互校；SRAT 自身已走完整张表（端点 = 表长）")
    print(f"  SRAT 内存段数 = {len(srat_mem)}（node0 有两段：低位 2GiB + 1.0TiB 处的 64GiB）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
