"""E2E 全链路差分正确性对拍（两进程，覆盖真实 Python 语法与模型使用场景）。

与 verify_against_aten.py 的差别：那个脚本只测 glue 入口 + 人工构造的 indices
列表（kernel 语义）；本脚本测 **真实 Python 语法 `x[...]`**（Python 归一化 →
视图/偏移/mask 转换/broadcast → dispatcher → 覆盖内核）整条 E2E 链路，并把
维度拉开：语法形态 × self 布局 × dtype × K/N/A 边界 × autograd × 线程数 ×
模型 mini-workload × 输出元数据/别名检查。

用法（两进程自动跑）：
  python tests/verify_e2e.py --mode both
或手动：
  HPC_INDEX_NO_OVERRIDE=1 python tests/verify_e2e.py --mode gen
  python tests/verify_e2e.py --mode check
"""
import argparse, os, pickle, subprocess, sys, torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REF_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "agent_space")
REF_FILE = os.path.join(REF_DIR, "verify_e2e_ref.pkl")


# ---------------------------------------------------------------------------
# 用例定义：每个用例 = (name, build, fn)
#   build(seed) -> inputs dict（两个进程各自以同 seed 重建，保证输入一致）
#   fn(inputs)  -> 输出（tensor 或 tensor 元组；两个进程用同一份代码）
# gen 进程：HPC_INDEX_NO_OVERRIDE=1，fn 走 true aten；check 进程：fn 走覆盖。
# ---------------------------------------------------------------------------

def _g(seed=0):
    return torch.Generator().manual_seed(1234 + seed)


def build_cases():
    cases = []

    def add(name, build, fn):
        cases.append((name, build, fn))

    # ============ G1 真实语法形态（E2E 模型代码里出现的写法） ============
    def b_basic(seed=0):
        g = _g(seed)
        return {"x": torch.randn(1000, 8, generator=g), "t": torch.randint(0, 1000, (2048,), generator=g)}
    add("g1_single_i64", b_basic, lambda i: i["x"][i["t"]])
    add("g1_single_i32", b_basic, lambda i: i["x"][i["t"].to(torch.int32)])

    def b_multi(seed=0):
        g = _g(seed)
        return {"x": torch.randn(200, 300, 5, generator=g),
                "a": torch.randint(0, 200, (50,), generator=g),
                "b": torch.randint(0, 300, (50,), generator=g)}
    add("g1_multi2", b_multi, lambda i: i["x"][i["a"], i["b"]])
    add("g1_multi2_tuple", b_multi, lambda i: i["x"][(i["a"], i["b"])])

    def b_multi3(seed=0):
        g = _g(seed)
        return {"x": torch.randn(30, 40, 50, 6, generator=g),
                "a": torch.randint(0, 30, (17,), generator=g),
                "b": torch.randint(0, 40, (17,), generator=g),
                "c": torch.randint(0, 50, (17,), generator=g)}
    add("g1_multi3", b_multi3, lambda i: i["x"][i["a"], i["b"], i["c"]])

    def b_newaxis(seed=0):
        g = _g(seed)
        return {"x": torch.randn(100, 4, generator=g), "t": torch.randint(0, 100, (64,), generator=g),
                "u": torch.randint(0, 4, (64,), generator=g)}
    add("g1_newaxis_after", b_newaxis, lambda i: i["x"][i["t"], None])
    add("g1_newaxis_before", b_newaxis, lambda i: i["x"][None, i["t"]])
    add("g1_newaxis_both", b_newaxis, lambda i: i["x"][None, i["t"], None])
    add("g1_ellipsis", b_newaxis, lambda i: i["x"][..., i["u"]])
    add("g1_ellipsis_newaxis", b_newaxis, lambda i: i["x"][..., i["u"], None])

    def b_slice_adv(seed=0):
        g = _g(seed)
        x = torch.randn(1000, 32, generator=g)
        return {"x": x, "trow": torch.randint(0, 490, (128,), generator=g),
                "tcol": torch.randint(0, 32, (128,), generator=g),
                "tfull": torch.randint(0, 1000, (128,), generator=g)}
    add("g1_slice_then_adv", b_slice_adv, lambda i: i["x"][10:500][i["trow"]])
    add("g1_slice_adv_mixed", b_slice_adv, lambda i: i["x"][10:500, i["tcol"]])
    add("g1_col_adv", b_slice_adv, lambda i: i["x"][:, i["tcol"]])
    add("g1_adv_then_slice", b_slice_adv, lambda i: i["x"][i["tfull"], 3:11])
    add("g1_strided_slice_adv", b_slice_adv, lambda i: i["x"][2:999:7][i["trow"] % 143])
    add("g1_int_then_adv", b_slice_adv, lambda i: i["x"][7][i["tcol"]])

    def b_mask(seed=0):
        g = _g(seed)
        x = torch.randn(500, 16, generator=g)
        m = torch.rand(500, generator=g) < 0.3
        m2 = torch.rand(500, 16, generator=g) < 0.5
        t = torch.randint(0, 16, (200,), generator=g)
        return {"x": x, "m": m, "m2": m2, "t": t,
                "mn": (torch.rand(1000, generator=g) < 0.3)[::2],
                "mcol": torch.rand(16, generator=g) < 0.5}
    add("g1_mask_1d", b_mask, lambda i: i["x"][i["m"]])
    add("g1_mask_2d", b_mask, lambda i: i["x"][i["m2"]])
    add("g1_mask_noncontig", b_mask, lambda i: i["x"][i["mn"]])
    add("g1_mask_plus_tensor", b_mask, lambda i: i["x"][i["m"], i["t"][:1]])
    add("g1_tensor_plus_mask", b_mask, lambda i: i["x"][i["t"][:1], i["mcol"]])

    def b_scalar(seed=0):
        g = _g(seed)
        return {"x": torch.randn(60, 70, generator=g),
                "s": torch.tensor(5), "v": torch.tensor(3),
                "e": torch.tensor([], dtype=torch.int64)}
    add("g1_0dim_index", b_scalar, lambda i: i["x"][i["s"]])
    add("g1_0dim_pair", b_scalar, lambda i: i["x"][i["s"], i["v"]])
    add("g1_empty_idx", b_scalar, lambda i: i["x"][i["e"]])

    def b_negdup(seed=0):
        g = _g(seed)
        return {"x": torch.randn(100, 5, generator=g),
                "t": torch.tensor([-1, -100, 0, 3, 3, 3, 99, -50, 0], dtype=torch.int64)}
    add("g1_neg_dup", b_negdup, lambda i: i["x"][i["t"]])

    def b_broadcast(seed=0):
        g = _g(seed)
        return {"x": torch.randn(200, 300, generator=g),
                "a": torch.randint(0, 200, (40, 1), generator=g),
                "b": torch.randint(0, 300, (1, 55), generator=g)}
    add("g1_broadcast_outer", b_broadcast, lambda i: i["x"][i["a"], i["b"]])
    add("g1_broadcast_rev", b_broadcast, lambda i: i["x"][i["a"].t(), i["b"].t()])

    def b_mix3(seed=0):
        g = _g(seed)
        return {"x": torch.randn(10, 20, 30, 5, generator=g),
                "a": torch.randint(0, 10, (6,), generator=g),
                "b": torch.randint(0, 20, (6,), generator=g),
                "c": torch.randint(0, 30, (6,), generator=g),
                "d": torch.randint(0, 5, (6,), generator=g)}
    add("g1_mid_slice_multi", b_mix3, lambda i: i["x"][i["a"], 5:15:3, i["b"]])
    add("g1_trailing_slice_multi", b_mix3, lambda i: i["x"][i["a"], i["b"], i["c"]])
    add("g1_ellipsis_mid", b_mix3, lambda i: i["x"][i["a"], ..., i["d"]])

    # ============ G2 self 布局（视图/偏移/非连续/channels_last/expand） ============
    def b_layout(seed=0):
        g = _g(seed)
        x = torch.randn(200, 64, generator=g)
        t = torch.randint(0, 64, (1024,), generator=g)
        return {"x": x, "t": t, "xt": x.t(), "xs": x[::2], "xsel": x[5],
                "xn": x[:, 10:50], "xboth": x[3:7:1, ::2], "t4": torch.randint(0, 4, (1024,), generator=g),
                "x3": torch.randn(8, 200, 64, generator=g),
                "cl": torch.randn(4, 32, 14, 14, generator=g).to(memory_format=torch.channels_last),
                "exp": torch.randn(200, 64, generator=g).unsqueeze(0).expand(3, -1, -1)}
    add("g2_transpose", b_layout, lambda i: i["xt"][i["t"]])
    add("g2_strided_self", b_layout, lambda i: i["xs"][i["t"]])
    add("g2_select_view_offset", b_layout, lambda i: i["xsel"][i["t"]])
    add("g2_narrow_last_dim", b_layout, lambda i: i["xn"][i["t"]])
    add("g2_narrow_both", b_layout, lambda i: i["xboth"][i["t4"]])
    add("g2_transpose3d", b_layout, lambda i: i["x3"].transpose(0, 1)[i["t"]])
    add("g2_channels_last", b_layout, lambda i: i["cl"][torch.arange(4)])
    add("g2_channels_last_adv", b_layout, lambda i: i["cl"][
        torch.arange(4).view(-1, 1, 1, 1), torch.arange(2).view(1, -1, 1, 1),
        torch.arange(14).view(1, 1, -1, 1), torch.arange(14).view(1, 1, 1, -1)])
    add("g2_expanded_self", b_layout, lambda i: i["exp"][torch.arange(3).unsqueeze(1), i["t"][:16]])

    def b_layout_raw(seed=0):
        g = _g(seed)
        x = torch.randn(20000, 16, generator=g)
        return {"x": x, "xs": x[::2], "t": torch.randint(0, 10000, (8192,), generator=g)}
    add("g2_strided_self_bigK", b_layout_raw, lambda i: i["xs"][i["t"]])

    def b_noncontig_idx(seed=0):
        g = _g(seed)
        t = torch.randint(0, 10000, (4096,), generator=g)
        return {"x": torch.randn(10000, 8, generator=g), "t": t[::2]}
    add("g2_noncontig_idx", b_noncontig_idx, lambda i: i["x"][i["t"]])

    # ============ G3 dtype / K / N / A 边界矩阵 ============
    def b_dtype(dt, K=77, N=500):
        def build(seed=0):
            g = _g(seed)
            x = torch.randn(N, 4, generator=g).to(dt)
            t = torch.randint(0, N, (K,), generator=g)
            return {"x": x, "t": t}
        return build
    for dt in (torch.float32, torch.float64, torch.float16, torch.bfloat16,
               torch.int64, torch.int32, torch.int16, torch.int8, torch.uint8,
               torch.bool, torch.complex64):
        add(f"g3_dtype_{str(dt).split('.')[-1]}", b_dtype(dt), lambda i: i["x"][i["t"]])

    def b_K(K, N=3000, A=4):
        def build(seed=0):
            g = _g(seed)
            return {"x": torch.randn(N, A, generator=g), "t": torch.randint(0, N, (K,), generator=g)}
        return build
    for K in (0, 1, 2, 3, 7, 64, 1023, 1024, 4095, 4096, 65535, 65536, 100001):
        add(f"g3_K{K}", b_K(K), lambda i: i["x"][i["t"]])

    def b_A(A, K=4096, N=2000):
        def build(seed=0):
            g = _g(seed)
            return {"x": torch.randn(N, A, generator=g), "t": torch.randint(0, N, (K,), generator=g)}
        return build
    for A in (1, 2, 3, 4, 16, 64, 256, 4096):
        add(f"g3_A{A}", b_A(A), lambda i: i["x"][i["t"]])

    def b_N1(seed=0):
        g = _g(seed)
        return {"x": torch.randn(1, 5, generator=g), "t": torch.zeros(100, dtype=torch.int64)}
    add("g3_N1", b_N1, lambda i: i["x"][i["t"]])

    def b_small_dom(seed=0):
        g = _g(seed)
        return {"x": torch.randn(2048, generator=g), "t": torch.randint(0, 2048, (50000,), generator=g)}
    add("g3_bucket_dom2048", b_small_dom, lambda i: i["x"][i["t"]])
    add("g3_bucket_dom2048_i32", b_small_dom, lambda i: i["x"][i["t"].to(torch.int32)])

    def b_mask_big(seed=0):
        g = _g(seed)
        x = torch.randn(1_000_000, generator=g)
        return {"x": x, "m": torch.rand(1_000_000, generator=g) < 0.3}
    add("g3_mask_1M", b_mask_big, lambda i: i["x"][i["m"]])

    # ============ G4 autograd E2E（前向 + 反向梯度差分） ============
    def b_grad(seed=0):
        g = _g(seed)
        x = torch.randn(500, 8, generator=g)
        return {"x": x.clone().requires_grad_(True), "t": torch.randint(0, 500, (2000,), generator=g),
                "ref": x}
    add("g4_grad_single", b_grad, lambda i: i["x"][i["t"]].sum().backward() or i["x"].grad)
    add("g4_grad_weighted", b_grad,
        lambda i: (i["x"][i["t"]] * torch.linspace(0.1, 2.0, 2000).unsqueeze(1)).sum().backward() or i["x"].grad)

    def b_grad_dup(seed=0):
        g = _g(seed)
        x = torch.randn(100, 4, generator=g)
        t = torch.randint(0, 100, (2000,), generator=g)
        return {"x": x.clone().requires_grad_(True), "t": t}
    add("g4_grad_dup_neg", b_grad_dup,
        lambda i: i["x"][i["t"] - 100].sum().backward() or i["x"].grad)

    def b_grad_mask(seed=0):
        g = _g(seed)
        x = torch.randn(1000, 4, generator=g)
        return {"x": x.clone().requires_grad_(True), "m": torch.rand(1000, generator=g) < 0.4}
    add("g4_grad_mask", b_grad_mask, lambda i: i["x"][i["m"]].sum().backward() or i["x"].grad)

    def b_grad_multi(seed=0):
        g = _g(seed)
        x = torch.randn(100, 90, generator=g)
        return {"x": x.clone().requires_grad_(True),
                "a": torch.randint(0, 100, (300,), generator=g),
                "b": torch.randint(0, 90, (300,), generator=g)}
    add("g4_grad_multi", b_grad_multi,
        lambda i: i["x"][i["a"], i["b"]].sum().backward() or i["x"].grad)

    def b_grad_f16(seed=0):
        g = _g(seed)
        x = torch.randn(300, 8, generator=g).to(torch.float16)
        return {"x": x.clone().requires_grad_(True), "t": torch.randint(0, 300, (777,), generator=g)}
    add("g4_grad_f16", b_grad_f16, lambda i: i["x"][i["t"]].float().sum().backward() or i["x"].grad)

    def _no_grad_fn(i):
        with torch.no_grad():
            return i["x"][i["t"]].clone()
    add("g4_no_grad_fwd", b_grad, _no_grad_fn)

    def _inf_mode_fn(i):
        with torch.inference_mode():
            return i["x"][i["t"]].clone()
    add("g4_inference_mode_fwd", b_grad, _inf_mode_fn)

    # ============ G5 输出布局（stride）矩阵：self 布局 × 索引形态 ============
    # aten::index 输出 stride 由 TensorIterator 置换恢复决定（非恒连续）：
    # channels_last/类 cl 置换 self 的行索引输出保持原布局；col-index/multi
    # 常为连续；连续 self 上的转置 2D 索引也可能非连续。值+shape+stride 全比对。
    def b_g5(seed=0):
        g = _g(seed)
        cl = torch.randn(4, 32, 14, 14, generator=g).to(memory_format=torch.channels_last)
        cl5 = torch.randn(2, 3, 4, 5, 6, generator=g).to(memory_format=torch.channels_last_3d)
        t3 = torch.randn(4, 5, 6, generator=g).transpose(1, 2)
        c4 = torch.randn(4, 32, 14, 14, generator=g)
        return {"cl": cl, "cl5": cl5, "t3": t3, "c4": c4,
                "rows4": torch.arange(4), "rows2": torch.arange(2),
                "tcol5": torch.arange(5), "u3": torch.arange(3),
                "a21": torch.arange(2).unsqueeze(1),
                "b13": torch.arange(3).unsqueeze(0),
                "t2d": torch.arange(6).reshape(2, 3) % 4,
                "t2dt": torch.arange(6).reshape(2, 3).t() % 4,
                "nn": torch.arange(20)[::2] % 4,
                "neg": torch.tensor([3, -1, 3, 0]),
                "mask4": torch.tensor([True, False, True, False])}
    add("g5_cl4d_row", b_g5, lambda i: i["cl"][i["rows4"]])
    add("g5_cl4d_row_noncontig", b_g5, lambda i: i["cl"][i["nn"]])
    add("g5_cl4d_row_negdup", b_g5, lambda i: i["cl"][i["neg"]])
    add("g5_cl4d_mask", b_g5, lambda i: i["cl"][i["mask4"]])
    add("g5_cl4d_2d_index", b_g5, lambda i: i["cl"][i["t2d"]])
    add("g5_cl4d_t2dt", b_g5, lambda i: i["cl"][i["t2dt"]])
    add("g5_cl4d_t_slice", b_g5, lambda i: i["cl"][i["rows2"], 1:3])
    add("g5_cl4d_newaxis", b_g5, lambda i: i["cl"][i["rows2"], None])
    add("g5_cl4d_col", b_g5, lambda i: i["cl"][:, i["tcol5"]])
    add("g5_cl4d_slice_adv", b_g5, lambda i: i["cl"][1:3, i["tcol5"]])
    add("g5_cl4d_multi2", b_g5, lambda i: i["cl"][i["a21"], i["b13"]])
    add("g5_cl4d_ellipsis_t", b_g5, lambda i: i["cl"][..., i["u3"]])
    add("g5_cl5d_row", b_g5, lambda i: i["cl5"][i["rows2"]])
    add("g5_cl5d_col", b_g5, lambda i: i["cl5"][:, i["rows2"]])
    add("g5_cl5d_multi2", b_g5, lambda i: i["cl5"][i["rows2"], i["a21"]])
    add("g5_t3d_row", b_g5, lambda i: i["t3"][i["rows2"]])
    add("g5_t3d_col", b_g5, lambda i: i["t3"][:, i["u3"]])
    add("g5_t3d_multi2", b_g5, lambda i: i["t3"][i["a21"], i["u3"]])
    add("g5_t3d_2d_index", b_g5, lambda i: i["t3"][torch.arange(4).reshape(2, 2)])
    add("g5_c4_row", b_g5, lambda i: i["c4"][i["rows4"]])
    add("g5_c4_col", b_g5, lambda i: i["c4"][:, i["tcol5"]])
    add("g5_c4_multi2", b_g5, lambda i: i["c4"][i["a21"], i["b13"]])
    add("g5_c4_t2dt", b_g5, lambda i: i["c4"][i["t2dt"]])
    add("g5_c4_ellipsis_t", b_g5, lambda i: i["c4"][..., i["u3"]])

    def b_g5b(seed=0):
        g = _g(seed)
        return {"x2": torch.randn(5, 6, generator=g),
                "x3": torch.randn(5, 6, 7, generator=g),
                "t2dt": torch.arange(6).reshape(2, 3).t() % 5,
                "rows3": torch.arange(3)}
    add("g5_c2d_t2dt", b_g5b, lambda i: i["x2"][i["t2dt"]])
    add("g5_c3d_t2dt", b_g5b, lambda i: i["x3"][i["t2dt"]])
    add("g5_2dt_row", b_g5b, lambda i: i["x2"].t()[i["rows3"]])
    add("g5_strided_row", b_g5b, lambda i: i["x3"][::2][i["rows3"]])
    add("g5_expand_row", b_g5b, lambda i: i["x2"].unsqueeze(0).expand(4, -1, -1)[i["rows3"]])

    # 大 K（>=1024）直写路径：上面 g5 用例全为小 K（走回退 tmp+copy_），
    # 这组才真正穿过 R10 rowcopy_direct（行索引 raw/linearize 两臂、mask、cl5d）。
    def b_g5big(seed=0):
        g = _g(seed)
        cl = torch.randn(64, 32, 7, 7, generator=g).to(memory_format=torch.channels_last)
        clraw = torch.randn(16, 32, 14, 14, generator=g).to(memory_format=torch.channels_last)
        clm = torch.randn(2048, 32, 3, 3, generator=g).to(memory_format=torch.channels_last)
        cl5 = torch.randn(2000, 3, 4, 5, 6, generator=g).to(memory_format=torch.channels_last_3d)
        m = torch.zeros(2048, dtype=torch.bool)
        m[torch.randperm(2048, generator=g)[:1500]] = True
        return {"cl": cl, "clraw": clraw, "clm": clm, "cl5": cl5, "m": m,
                "rows": torch.randint(0, 64, (1500,), generator=g),
                "rowsr": torch.randint(0, 16, (1500,), generator=g),
                "rows5": torch.randint(0, 2000, (1500,), generator=g)}
    add("g5big_cl4d_row", b_g5big, lambda i: i["cl"][i["rows"]])            # A=1568 linearize 臂
    add("g5big_cl4d_row_raw", b_g5big, lambda i: i["clraw"][i["rowsr"]])    # A=6272 raw 臂
    add("g5big_cl4d_mask", b_g5big, lambda i: i["clm"][i["m"]])            # mask 展开族 K=1500
    add("g5big_cl5d_row", b_g5big, lambda i: i["cl5"][i["rows5"]])         # cl3d 布局 A=360

    # ============ G6 E2E 模型 mini-workload（前向+反向全对拍） ============
    def m_spconv(seed=0):
        g = _g(seed)
        N, C, nb, nk = 2000, 32, 27, 64
        feat = torch.randn(N, C, generator=g)
        w = torch.randn(C, C, 3, 3, generator=g)
        rule = torch.randint(0, N, (nb, nk), generator=g)
        return {"feat": feat.clone().requires_grad_(True), "w": w.clone().requires_grad_(True),
                "rule": rule.reshape(-1)}
    def f_spconv(i):
        y = i["feat"][i["rule"]].view(27, 64, 32)
        out = torch.nn.functional.conv2d(y.permute(0, 2, 1).unsqueeze(-1),
                                         i["w"].sum(dim=(2, 3)).unsqueeze(-1).unsqueeze(-1))
        loss = out.square().mean()
        loss.backward()
        return (loss.detach(), i["feat"].grad.clone())
    add("g6_spconv_step", m_spconv, f_spconv)

    def m_msgpass(seed=0):
        g = _g(seed)
        N, C, E = 5000, 16, 30000
        h = torch.randn(N, C, generator=g)
        src = torch.randint(0, N, (E,), generator=g)
        return {"h": h.clone().requires_grad_(True), "src": src}
    def f_msgpass(i):
        agg = i["h"][i["src"]].view(-1, 100, 16).mean(dim=1)   # 消息聚合
        loss = agg.square().sum()
        loss.backward()
        return (loss.detach(), i["h"].grad.clone())
    add("g6_msgpass_step", m_msgpass, f_msgpass)

    def m_embed(seed=0):
        g = _g(seed)
        V, C, B = 10000, 64, 2048
        emb = torch.randn(V, C, generator=g)
        ids = torch.randint(0, V, (B,), generator=g)
        lin = torch.randn(C, 8, generator=g)
        return {"emb": emb.clone().requires_grad_(True), "ids": ids, "lin": lin}
    def f_embed(i):
        e = i["emb"][i["ids"]]
        loss = (e @ i["lin"]).square().mean()
        loss.backward()
        return (loss.detach(), i["emb"].grad.clone())
    add("g6_embed_step", m_embed, f_embed)

    def m_mask_batch(seed=0):
        g = _g(seed)
        N, C = 30000, 8
        x = torch.randn(N, C, generator=g)
        lin = torch.randn(C, 4, generator=g)
        return {"x": x.clone().requires_grad_(True),
                "m": torch.rand(N, generator=g) < 0.7, "lin": lin}
    def f_mask_batch(i):
        sel = i["x"][i["m"]]
        loss = (sel @ i["lin"]).square().mean()
        loss.backward()
        return (loss.detach(), i["x"].grad.clone(), sel.shape)
    add("g6_mask_batch_step", m_mask_batch, f_mask_batch)

    def m_w_gather(seed=0):
        g = _g(seed)
        K, C, O, E = 27, 64, 64, 5000
        w = torch.randn(K, C, O, generator=g)
        ki = torch.randint(0, K, (E,), generator=g)
        return {"w": w.clone().requires_grad_(True), "ki": ki}
    def f_w_gather(i):
        y = i["w"][i["ki"]]                       # M1 权重块
        loss = y.square().mean()
        loss.backward()
        return (loss.detach(), i["w"].grad.clone())
    add("g6_weight_block_step", m_w_gather, f_w_gather)

    def m_fps(seed=0):
        g = _g(seed)
        return {"pos": torch.randn(4096, 3, generator=g)}
    def f_fps(i):
        # FPS/ball_query 风格：循环里逐个取行（0-dim 张量下标 → aten::index 路径）
        p = i["pos"]
        N = p.shape[0]
        g = _g(99)
        sel = torch.empty(64, 3)
        cur = 0
        for j in range(64):
            sel[j] = p[torch.tensor(cur)]
            cur = int(torch.randint(0, N, (1,), generator=g))
        return sel.clone()
    add("g6_fps_loop", m_fps, f_fps)

    return cases


# ---------------------------------------------------------------------------
# 运行器
# ---------------------------------------------------------------------------

def run_gen(args):
    os.makedirs(REF_DIR, exist_ok=True)
    refs = {}
    for name, build, fn in build_cases():
        out = fn(build())
        if isinstance(out, torch.Tensor):
            out = (out,)
        refs[name] = tuple(t.detach().clone() if torch.is_tensor(t) else t for t in out)
        shapes = [tuple(t.shape) for t in refs[name] if torch.is_tensor(t)]
        print(f"gen {name}: {shapes}")
    with open(REF_FILE, "wb") as f:
        pickle.dump(refs, f)
    print(f"wrote {len(refs)} cases -> {REF_FILE}")


def _describe(t):
    if torch.is_tensor(t):
        return f"shape={tuple(t.shape)} dtype={t.dtype} stride={t.stride()} contig={t.is_contiguous()}"
    return repr(t)


def run_check(args):
    import index_sve  # noqa: F401  覆盖激活
    with open(REF_FILE, "rb") as f:
        refs = pickle.load(f)
    failed = []
    for name, build, fn in build_cases():
        if name not in refs:
            print(f"SKIP {name}: 无参考")
            continue
        ref = refs[name]
        try:
            out = fn(build())
            if isinstance(out, torch.Tensor):
                out = (out,)
            out = tuple(t.detach().clone() if torch.is_tensor(t) else t for t in out)
            assert len(out) == len(ref), f"输出个数 {len(out)} != 参考 {len(ref)}"
            for k, (y, r) in enumerate(zip(out, ref)):
                if not torch.is_tensor(r):
                    assert y == r, f"out[{k}] {y!r} != {r!r}"
                    continue
                assert torch.is_tensor(y), f"out[{k}] 非张量"
                assert y.shape == r.shape, f"out[{k}] shape {tuple(y.shape)} != {tuple(r.shape)}"
                assert y.dtype == r.dtype, f"out[{k}] dtype {y.dtype} != {r.dtype}"
                assert y.stride() == r.stride(), f"out[{k}] stride {y.stride()} != {r.stride()}"
                if y.numel():
                    torch.testing.assert_close(y, r, equal_nan=True)
            print(f"PASS {name}")
        except Exception as e:
            failed.append(name)
            print(f"FAIL {name}: {e!r}")
    print(f"== {len(refs) - len(failed)}/{len(refs)} passed ==")
    if failed:
        print("FAILED:", ", ".join(failed))
    sys.exit(1 if failed else 0)


def run_threads_matrix(args):
    """并行正确性：同一 case 在 1/4/38 线程下结果必须一致（race 检测）。"""
    import index_sve  # noqa: F401
    big = [c for c in build_cases() if c[0].startswith(("g3_", "g6_", "g5big_"))]
    bad = []
    for name, build, fn in big:
        outs = []
        for nt in (1, 4, 38):
            torch.set_num_threads(nt)
            out = fn(build())
            outs.append(out.detach().clone() if torch.is_tensor(out) else out)
        same = all(torch.equal(outs[0], o) for o in outs[1:] if torch.is_tensor(o))
        if not same:
            bad.append(name)
            print(f"RACE-DIFF {name}: 1t 与多线程结果不一致")
        else:
            print(f"THREAD-OK {name}")
    torch.set_num_threads(38)
    print(f"== threads matrix: {'ALL OK' if not bad else 'DIFF: ' + ', '.join(bad)} ==")
    sys.exit(1 if bad else 0)


def run_both(args):
    env = dict(os.environ)
    env["HPC_INDEX_NO_OVERRIDE"] = "1"
    gen = subprocess.run([sys.executable, __file__, "--mode", "gen"], env=env,
                         capture_output=True, text=True)
    print(gen.stdout, end="")
    if gen.returncode:
        print(gen.stderr, file=sys.stderr)
        sys.exit(gen.returncode)
    check = subprocess.run([sys.executable, __file__, "--mode", "check"],
                           capture_output=True, text=True)
    print(check.stdout, end="")
    if check.stderr:
        print(check.stderr, file=sys.stderr)
    rc = check.returncode
    if rc == 0 and args.threads:
        tm = subprocess.run([sys.executable, __file__, "--mode", "threads"],
                            capture_output=True, text=True)
        print(tm.stdout, end="")
        if tm.stderr:
            print(tm.stderr, file=sys.stderr)
        rc = tm.returncode
    sys.exit(rc)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("gen", "check", "both", "threads"), default="both")
    p.add_argument("--threads", action="store_true", help="both 模式追加线程一致性矩阵")
    args = p.parse_args()
    {"gen": run_gen, "check": run_check, "threads": run_threads_matrix,
     "both": run_both}[args.mode](args)
