#!/usr/bin/env python3
"""解析 HMAT / SLIT / SRAT，供内存拓扑与固件声明的带宽/延迟等级对照。

用法: python3 tools/parse-acpi-hw.py <目录，含 HMAT/SLIT/SRAT 三个原始表>

表结构按 ACPI 6.4（HMAT 见 §5.4.4 的三张子表）。未知布局不猜：打印长度与十六进制
前 96 字节，让读的人自己按规范对。
"""
import os
import struct
import sys

HMAT_TYPES = {0: "地址区间", 1: "延迟/带宽等级", 2: "内存侧缓存"}
DTYPES = {0: "访问延迟", 1: "读带宽", 2: "写带宽"}
SRAT_TYPES = {0: "处理器 LAPIC", 1: "内存亲和", 2: "中断", 5: "通用发起者"}


def read(path):
    with open(path, "rb") as f:
        return f.read()


def header(data):
    sig, length, rev, csum, oemid, oemtid, oemrev, crid, crrev = struct.unpack_from(
        "<4sIBB6s8sI4sI", data, 0
    )
    return {
        "sig": sig.decode(errors="replace"),
        "length": length,
        "rev": rev,
        "file": len(data),
        "oem": oemid.decode(errors="replace").strip("\x00 "),
        "table_id": oemtid.decode(errors="replace").strip("\x00 "),
    }


def subtable_iter(data, start=36):
    """HMAT 子表头是 type(u16)+reserved(u16)+length(u32)，共 8 字节。"""
    off = start
    while off + 8 <= min(len(data), header(data)["length"]):
        typ, _resv, length = struct.unpack_from("<HHI", data, off)
        if length < 8 or off + length > len(data):
            yield typ, off, length, data[off:]
            return
        yield typ, off, length, data[off : off + length]
        off += length


def srat_iter(data, start=36):
    """SRAT 子表头是 type(u8)+length(u8)，共 2 字节。"""
    off = start
    while off + 2 <= min(len(data), header(data)["length"]):
        typ, length = data[off], data[off + 1]
        if length < 2 or off + length > len(data):
            yield typ, off, length, data[off:]
            return
        yield typ, off, length, data[off : off + length]
        off += length


def hx(b, n=96):
    return " ".join(f"{x:02x}" for x in b[:n])


def parse_hmat(data):
    h = header(data)
    subs = list(subtable_iter(data))
    counts = {}
    for typ, *_ in subs:
        counts[typ] = counts.get(typ, 0) + 1
    print("=" * 78)
    print(f"[HMAT] 签名={h['sig']} 版本={h['rev']} 长度={h['length']} 文件={h['file']}")
    print(f"       OEM={h['oem']} 表名={h['table_id']}")
    print("       子表：" + "、".join(f"{HMAT_TYPES.get(t, t)}×{c}" for t, c in sorted(counts.items())))

    ranges = []
    locality = []
    caches = []
    for typ, off, length, body in subs:
        if typ == 0:
            if length < 40:
                print(f"  ! 地址区间子表长度 {length} < 40，跳过解码（原始：{hx(body)}）")
                continue
            flags = struct.unpack_from("<H", body, 8)[0]
            ppd = struct.unpack_from("<I", body, 12)[0]
            mpd = struct.unpack_from("<I", body, 16)[0]
            if length >= 48:
                base, mlen = struct.unpack_from("<QQ", body, 32)
            else:
                base, mlen = struct.unpack_from("<QQ", body, 24)
            ranges.append((ppd, mpd, base, mlen, flags, length, off, body))
        elif typ == 1:
            locality.append((off, length, body))
        elif typ == 2:
            caches.append((off, length, body))
        else:
            print(f"  ! 未知子表类型 {typ}，长度 {length}，偏移 {off}，原始：{hx(body)}")

    if ranges:
        print(f"\n  地址区间子表（{len(ranges)} 条；flags bit0=启用 bit1=可热插拔）：")
        print("    proc_pd  mem_pd        base(hex)         length(hex)   size     flags  子表长")
        for ppd, mpd, base, mlen, flags, length, off, _b in ranges:
            print(
                f"    {ppd:>7}  {mpd:>6}  0x{base:012x}  0x{mlen:012x}  "
                f"{mlen / 2**30:6.1f}GiB  0x{flags:04x}  {length}"
            )

    for idx, (off, length, body) in enumerate(locality, 1):
        flags = body[8]
        dtype = body[9]
        num_i, num_t, base_unit = struct.unpack_from("<III", body, 12)
        need = 24 + 4 * (num_i + num_t) + 8 * num_i * num_t
        print(
            f"\n  [等级表 #{idx}] 数据类型={DTYPES.get(dtype, dtype)} 基本单位={base_unit} "
            f"发起域={num_i} 目标域={num_t} 子表长={length}（按布局需要 {need}）"
        )
        print(f"    flags=0x{flags:02x}  原始前 96 字节：{hx(body)}")
        if need != length:
            print("    ! 长度与 ACPI 6.4 布局算出的不一致；只打印原始表头，不做解码")
            continue
        init = struct.unpack_from(f"<{num_i}I", body, 24)
        tgt = struct.unpack_from(f"<{num_t}I", body, 24 + 4 * num_i)
        vals = struct.unpack_from(f"<{2 * num_i * num_t}I", body, 24 + 4 * (num_i + num_t))
        print(f"    发起域列表={list(init)}")
        print(f"    目标域列表={list(tgt)}")
        unit = base_unit / 1000.0
        label = "ns" if dtype == 0 else "GB/s"
        print(f"    单位换算：条目 × {base_unit} {'ps' if dtype == 0 else 'MB/s'} → {label}")
        show = [i for i, t in enumerate(tgt)]
        print("      发起\\目标 " + " ".join(f"{tgt[i]:>7}" for i in show))
        for a in range(num_i):
            row = vals[a * num_t * 2 : (a + 1) * num_t * 2]
            cells = []
            for bi in show:
                lat, bw = row[2 * bi], row[2 * bi + 1]
                v = lat if dtype == 0 else bw
                cells.append(f"{v * unit:7.1f}")
            print(f"      {init[a]:>8}  " + " ".join(cells))

    for idx, (off, length, body) in enumerate(caches, 1):
        if length < 32:
            print(f"\n  [内存侧缓存 #{idx}] 长度 {length} < 32，原始：{hx(body)}")
            continue
        pd, csize, attrs, nhandle = struct.unpack_from("<IxxxxQIH", body, 8)
        print(
            f"\n  [内存侧缓存 #{idx}] 内存域={pd} 容量={csize / 2**30:.1f}GiB "
            f"属性=0x{attrs:08x} SMBIOS句柄数={nhandle}"
        )
        print(f"    level={attrs & 0xF} 关联度={(attrs >> 4) & 0x3} 写策略={(attrs >> 6) & 0x3} "
              f"类型={(attrs >> 8) & 0x3}  原始：{hx(body)}")
    return ranges


def parse_slit(data):
    h = header(data)
    n = struct.unpack_from("<Q", data, 36)[0]
    mat = data[44 : 44 + n * n]
    print("\n" + "=" * 78)
    print(f"[SLIT] 版本={h['rev']} 长度={h['length']} 文件={h['file']} 局部数={n}")
    if len(mat) != n * n:
        print(f"  ! 矩阵长度不符：需要 {n * n}，实际 {len(mat)}")
        return None
    m = [list(mat[i * n : (i + 1) * n]) for i in range(n)]
    print("  矩阵（行=发起 node，列=目标 node 0..31）：")
    print("      " + " ".join(f"{c:>2}" for c in range(n)))
    for r in range(n):
        print(f"  {r:>3} " + " ".join(f"{v:>2}" for v in m[r]))
    return m


def parse_srat(data):
    h = header(data)
    mem, gen, other = [], [], {}
    for typ, off, length, body in srat_iter(data, 36):
        if typ == 1:
            pd = struct.unpack_from("<I", body, 2)[0]
            base = struct.unpack_from("<Q", body, 8)[0]
            mlen = struct.unpack_from("<Q", body, 16)[0]
            flags = struct.unpack_from("<I", body, 28)[0]
            mem.append((pd, base, mlen, flags, length))
        elif typ == 5:
            handle = body[8:24].rstrip(b"\x00")
            pd = struct.unpack_from("<I", body, 4)[0]
            gen.append((pd, handle, length))
        else:
            other[typ] = other.get(typ, 0) + 1
    print("\n" + "=" * 78)
    print(f"[SRAT] 版本={h['rev']} 长度={h['length']} 文件={h['file']}")
    print("  子表计数：" + "、".join(f"{SRAT_TYPES.get(t, t)}×{c}" for t, c in sorted(other.items())) +
          f"、内存亲和×{len(mem)}、通用发起者×{len(gen)}")
    print("  内存亲和（pd, base, size, flags：bit0=启用 bit1=可热插拔）：")
    for pd, base, mlen, flags, length in mem:
        print(f"    pd={pd:>2}  0x{base:012x}  {mlen / 2**30:7.1f}GiB  flags=0x{flags:02x}  子表长={length}")
    if gen:
        print("  通用发起者（pd, 设备句柄）：")
        for pd, handle, length in gen:
            print(f"    pd={pd:>2}  handle={handle!r}  子表长={length}")
    return mem


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "."
    print(f"# ACPI 表解析（目录 {os.path.abspath(d)}）")
    paths = {n: os.path.join(d, n) for n in ("HMAT", "SLIT", "SRAT")}
    missing = [n for n, p in paths.items() if not os.path.exists(p)]
    if missing:
        print("缺少表：" + "、".join(missing) + "（需要 root 才能读 /sys/firmware/acpi/tables）")
        return 1

    hmat_ranges = parse_hmat(read(paths["HMAT"]))
    slit = parse_slit(read(paths["SLIT"]))
    srat_mem = parse_srat(read(paths["SRAT"]))

    print("\n" + "=" * 78)
    print("[交叉校验] HMAT 地址区间 vs SRAT 内存亲和")
    srat_set = {(base, mlen) for _pd, base, mlen, _f, _l in srat_mem}
    ok = sum(1 for _p, _m, base, mlen, *_ in hmat_ranges if (base, mlen) in srat_set)
    print(f"  HMAT {len(hmat_ranges)} 条中与 SRAT 完全一致的有 {ok} 条"
          + ("（布局解析正确）" if ok == len(hmat_ranges) else "（不一致 → 上面 HMAT 偏移可能猜错，看原始 hex）"))

    if slit:
        def d(a, b):
            return slit[a][b]
        print("\n[SLIT 读数] 距离分组")
        for a in (5, 13):
            grp = {}
            for b in range(16, 32):
                grp.setdefault(d(a, b), []).append(b)
            print(f"  node{a} → " + "；".join(f"d={k}: {v}" for k, v in sorted(grp.items())))
        print(f"  d(17,18)={d(17, 18)} d(17,24)={d(17, 24)} d(5,13)={d(5, 13)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
