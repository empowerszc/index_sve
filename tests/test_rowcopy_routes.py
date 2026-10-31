import os, sys, torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import index_sve
from index_sve import _hpc_index as C

if not index_sve.OVERRIDE_ACTIVE:
    raise SystemExit(
        "test_rowcopy_routes 需要覆盖激活：cl4d 六项靠 x[...] 验证路由，覆盖关闭时 "
        "2 项必挂、4 项假通过（HPC_INDEX_NO_OVERRIDE=1？）")


def test_small_row_raw_route():
    x = torch.randn(20000, 8)
    idx = torch.randint(0, 20000, (8192,), dtype=torch.int32)
    ref = torch.index_select(x, 0, idx)
    old = torch.get_num_threads()
    try:
        torch.set_num_threads(8)
        before = C.rowcopy_raw_count()
        y = index_sve.index(x, [idx])
        torch.testing.assert_close(y, ref)
        assert C.rowcopy_raw_count() == before + 1, "A=8 multi-thread did not skip linearize"
    finally:
        torch.set_num_threads(old)


def test_multi_index_small_route():
    x = torch.randn(20000, 8, 64)
    i = torch.randint(0, 20000, (8192,))
    j = torch.randint(0, 8, (8192,))
    ref = torch.index_select(x.reshape(-1, 64), 0, i * x.size(1) + j)
    before = C.small_multi_count()
    y = index_sve.index(x, [i, j])
    torch.testing.assert_close(y, ref)
    assert C.small_multi_count() == before + 1, "small multi-index did not take the direct path"


def test_multi_index_raw_route():
    x = torch.randn(20000, 8, 64)
    i = torch.randint(0, 20000, (131072,))
    j = torch.randint(0, 8, (131072,))
    ref = torch.index_select(x.reshape(-1, 64), 0, i * x.size(1) + j)
    old = torch.get_num_threads()
    try:
        for t in (16, 36):
            torch.set_num_threads(t)
            before = C.rowcopy_raw_count()
            y = index_sve.index(x, [i, j])
            torch.testing.assert_close(y, ref)
            assert C.rowcopy_raw_count() == before + 1, f"multi-index A=64 t={t} did not skip linearize"
        torch.set_num_threads(8)
        before = C.rowcopy_raw_count()
        index_sve.index(x, [i, j])
        assert C.rowcopy_raw_count() == before, "multi-index t=8 should keep linearize"
    finally:
        torch.set_num_threads(old)


def test_grain_floor():
    old = torch.get_num_threads()
    try:
        torch.set_num_threads(32)
        assert C.debug_grain(100) == 64
        assert C.debug_grain(1_000_000) == 1_000_000 // (32 * 8)
    finally:
        torch.set_num_threads(old)


def test_medium_a_raw_route():
    # T15-E：2D A=16 K=8192 曾落在 raw 路由空档（A<16 或 A>=4096），32/36t 0.73/0.76x。
    # 放宽到 A<=64 后，单索引 A=16 全线程跳过 linearize。
    x = torch.randn(20000, 16)
    idx = torch.randint(0, 20000, (8192,), dtype=torch.int32)
    ref = torch.index_select(x, 0, idx)
    old = torch.get_num_threads()
    try:
        for t in (8, 36):
            torch.set_num_threads(t)
            before = C.rowcopy_raw_count()
            y = index_sve.index(x, [idx])
            torch.testing.assert_close(y, ref)
            assert C.rowcopy_raw_count() == before + 1, f"A=16 t={t} did not skip linearize"
    finally:
        torch.set_num_threads(old)


def test_medium_a_large_k_keeps_linearize():
    # T15 复调：A<=64 的 raw 放行只限 B*K<=65536。大 K 下 linearize 的预校验偏移 +
    # 无检查拷贝更快（A/B 实测 A=16 K=200k 8t raw 慢 32%、A=64 E=1M 8t 慢 30%）。
    x = torch.randn(20000, 64)
    idx = torch.randint(0, 20000, (200_000,), dtype=torch.int32)
    ref = torch.index_select(x, 0, idx)
    old = torch.get_num_threads()
    try:
        torch.set_num_threads(8)
        before = C.rowcopy_raw_count()
        y = index_sve.index(x, [idx])
        torch.testing.assert_close(y, ref)
        assert C.rowcopy_raw_count() == before, "A=64 K=200k should keep linearize"
    finally:
        torch.set_num_threads(old)


def test_f32_i64_1d_route():
    # T15-C：f32 数据 + int64 下标的 1D 纯 gather。小源（≤512KB，L1/L2 常驻）标量
    # 单遍更快——4B SVE gather 微码吞吐下限 ~1.2ns/元素，高于标量 L1 命中循环
    # ~0.63ns/元素（实测 K=65536 源 16KB：SVE 75µs vs 标量 41µs）；大源（L3/DRAM）
    # 走专用单遍 SVE 内核（gather MLP 吸收延迟，K=10M 源 8MB：8t 1.32→1.93x）。
    # 不能经 linearize 落 gather_index——bucket 扫描+linearize+bucket 扫描+SVE
    # 共 4 遍 fork-join（实测 K=65536 56→110µs）。
    idx = torch.randint(0, 4096, (65536,), dtype=torch.int64)
    x = torch.randn(4096)
    ref = torch.index_select(x, 0, idx)
    big_x = torch.randn(2_000_000)
    big_idx = torch.randint(0, 2_000_000, (100_000,), dtype=torch.int64)
    big_ref = torch.index_select(big_x, 0, big_idx)
    old = torch.get_num_threads()
    try:
        torch.set_num_threads(16)
        before = C.single_1d_count()
        before_s = C.single_1d_sve_count()
        # 小源：标量单遍
        torch.testing.assert_close(index_sve.index(x, [idx]), ref)
        assert C.single_1d_count() == before + 1, "small-source f32+i64 should stay scalar"
        assert C.single_1d_sve_count() == before_s, "small source must not use SVE kernel"
        # 大源：直连 SVE 单遍
        torch.testing.assert_close(index_sve.index(big_x, [big_idx]), big_ref)
        assert C.single_1d_sve_count() == before_s + 1, "large-source f32+i64 should hit SVE kernel"
        assert C.single_1d_count() == before + 1, "large source must not be scalar"
        # 对照：complex64 仍无 SVE 内核，继续走标量单遍（计数器有效性）
        cx = torch.randn(4096, dtype=torch.complex64)
        cref = torch.index_select(cx, 0, idx)
        torch.testing.assert_close(index_sve.index(cx, [idx]), cref)
        assert C.single_1d_count() == before + 2, "complex64 control should stay scalar"
    finally:
        torch.set_num_threads(old)


def test_small_single_thread_scaled_threshold():
    # T15：small_single 并行阈值按线程缩放——38t 下 K=1024×A=16（16384 元素）的
    # fork-join 净亏 ~17µs，保持串行；K=4096×A=16 仍并行；8t 行为不变。
    old = torch.get_num_threads()
    try:
        torch.set_num_threads(38)
        assert not C.debug_ss_parallel(1024, 16), "K=1024 A=16 should stay serial at 38t"
        assert C.debug_ss_parallel(4096, 16), "K=4096 A=16 should parallelize at 38t"
        torch.set_num_threads(8)
        assert C.debug_ss_parallel(1024, 16), "K=1024 A=16 should parallelize at 8t"
    finally:
        torch.set_num_threads(old)


def test_cl4d_row_index_routes_rowcopy_raw():
    # R10：cl self 行索引的输出保持 cl（aten 语义），且源/出尾块同为 dense 置换
    # （(C,H,W)/(1,WC,C)），行块 memcpy 语义等价——直写 out，消除 tmp+copy_。
    # 修复前走 gather_fused 进 tmp（cl 的 col_off 全为 +32 跨步、max_run=1，
    # strided_rows 不接管；计数器不增）。1t/8t 双档钉住路由：直写守卫在
    # dispatch 顶部、线程数无关，防后续守卫重排引入 1t/多线程分叉。
    x = torch.randn(4096, 32, 14, 14).to(memory_format=torch.channels_last)
    rows = torch.randint(0, 4096, (2048,), dtype=torch.int64)
    xc = x.contiguous()
    ref = xc[rows]
    old = torch.get_num_threads()
    try:
        for nt in (1, 8):
            torch.set_num_threads(nt)
            before = C.rowcopy_raw_count()
            y = x[rows]
            torch.testing.assert_close(y, ref)
            assert y.is_contiguous(memory_format=torch.channels_last), y.stride()
            assert y.stride() == (6272, 1, 448, 32), y.stride()
            assert C.rowcopy_raw_count() == before + 1, \
                f"nt={nt}: cl4d row index should block-copy direct into cl out"
    finally:
        torch.set_num_threads(old)


def test_cl4d_mask_routes_rowcopy_raw():
    # R10 mask 族：mask 覆盖 dim0、K>=1024 → normalize 展开为单索引，同直写路由。
    x = torch.randn(4096, 32, 14, 14).to(memory_format=torch.channels_last)
    m = torch.zeros(4096, dtype=torch.bool)
    m[torch.randperm(4096)[:2048]] = True
    xc = x.contiguous()
    ref = xc[m]
    old = torch.get_num_threads()
    try:
        torch.set_num_threads(8)
        before = C.rowcopy_raw_count()
        y = x[m]
        torch.testing.assert_close(y, ref)
        assert y.stride() == (6272, 1, 448, 32), y.stride()
        assert C.rowcopy_raw_count() == before + 1, "cl4d mask should block-copy direct into cl out"
    finally:
        torch.set_num_threads(old)


def test_cl3d_perm_row_index_direct_values():
    # R10 linearize 臂：A=3072 落在 [64,4096)（raw 形态条件不满足）→
    # linearize+gather_index 的 rowcopy_sve 直写。值/stride 断言即可钉住
    # 路由正确性（误入 col_off 类路径会块内错位、值被打乱）。
    x = torch.randn(4096, 96, 32).permute(0, 2, 1)  # (4096,32,96) strides (3072,1,32)
    rows = torch.randint(0, 4096, (2048,), dtype=torch.int64)
    ref = x.contiguous()[rows]
    old = torch.get_num_threads()
    try:
        torch.set_num_threads(8)
        y = x[rows]
        torch.testing.assert_close(y, ref)
        assert y.stride() == (3072, 1, 32), y.stride()
    finally:
        torch.set_num_threads(old)


def test_cl4d_sliced_tail_excluded():
    # 排除：尾块被切片（w∈[1,3) stride 32 有洞）非 dense 置换 → 不可整行 memcpy，
    # 必须回退 tmp+copy_（计数器不增）。K/A/线程均满足，唯一排除原因是尾块非 dense。
    x = torch.randn(4096, 32, 14, 14).to(memory_format=torch.channels_last)
    t = torch.randint(0, 4096, (2048,), dtype=torch.int64)
    xc = x.contiguous()
    ref = xc[t, 1:3]
    old = torch.get_num_threads()
    try:
        torch.set_num_threads(8)
        before = C.rowcopy_raw_count()
        y = x[t, 1:3]
        torch.testing.assert_close(y, ref)
        assert C.rowcopy_raw_count() == before, "sliced cl tail is not dense; must not block-copy direct"
    finally:
        torch.set_num_threads(old)


def test_cl4d_small_k_excluded():
    # 排除：K<1024（小张量路径域）不参与直写资格。
    x = torch.randn(4096, 32, 14, 14).to(memory_format=torch.channels_last)
    rows = torch.randint(0, 4096, (512,), dtype=torch.int64)
    xc = x.contiguous()
    ref = xc[rows]
    old = torch.get_num_threads()
    try:
        torch.set_num_threads(8)
        before = C.rowcopy_raw_count()
        y = x[rows]
        torch.testing.assert_close(y, ref)
        assert C.rowcopy_raw_count() == before, "K<1024 stays on small-tensor path"
    finally:
        torch.set_num_threads(old)


def test_cl4d_half_excluded():
    # 排除：非 f32/f64 dtype（rowcopy_sve/linearize 臂 dtype 集）不参与直写。
    x = torch.randn(4096, 32, 14, 14).to(memory_format=torch.channels_last).half()
    rows = torch.randint(0, 4096, (2048,), dtype=torch.int64)
    xc = x.contiguous()
    ref = xc[rows]
    old = torch.get_num_threads()
    try:
        torch.set_num_threads(8)
        before = C.rowcopy_raw_count()
        y = x[rows]
        torch.testing.assert_close(y, ref)
        assert C.rowcopy_raw_count() == before, "f16 cl self must not use direct rowcopy"
    finally:
        torch.set_num_threads(old)


if __name__ == "__main__":
    sys.path.insert(0, __file__.rsplit("/", 1)[0])
    import _harness
    _harness.main(globals())
