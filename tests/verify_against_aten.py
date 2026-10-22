"""两进程 true-aten 值级对拍脚本（他人/agent 验证正确性用）。

覆盖激活后同一进程内 `x[i]` 也走本算子（覆盖下的对拍陷阱，见 docs/semantics.md
§6.2），因此值级真对拍必须两进程：

1) 生成参考（无覆盖进程，走 true aten）：
   HPC_INDEX_NO_OVERRIDE=1 python tests/verify_against_aten.py --mode gen
2) 校验（覆盖进程，比对本算子输出 vs 参考）：
   python tests/verify_against_aten.py --mode check

两模式基于同一 shape 矩阵（`--shape-set` 可挑子集：small / large / all）。
large 集含 K>=1024 用例，覆盖 linearize+gather_index / SVE / 桶化 / M1 raw /
T5 单遍各路径。参考文件默认写 `agent_space/`（git-ignored），可用 --ref-dir 改。

用法示例：
  HPC_INDEX_NO_OVERRIDE=1 python tests/verify_against_aten.py --mode gen --shape-set all
  python tests/verify_against_aten.py --mode check --shape-set all
  # 两进程自动跑（一个无覆盖生成 .pt，一个覆盖比对）：
  python tests/verify_against_aten.py --mode both
"""
import argparse, os, pickle, sys, torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REF_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "agent_space")
REF_FILE = os.path.join(REF_DIR, "verify_against_aten_ref.pkl")


def gen_cases(shape_set):
    """返回 [(name, self, indices)] 的生成器（在无覆盖进程中调用，true aten 语义）。"""
    rng = torch.Generator().manual_seed(12345)
    cases = []
    def add(name, x, idx):
        cases.append((name, x, [t.clone() if torch.is_tensor(t) and t is not None else t for t in idx]))

    if shape_set in ("small", "all"):
        add("1d_small_f32", torch.randn(10, generator=rng), [torch.tensor([1, 3, 0, 3, 9])])
        add("1d_neg", torch.arange(10, dtype=torch.float32), [torch.tensor([-1, -10, 0])])
        add("row_contig", torch.randn(8, 5, generator=rng), [None, torch.tensor([1, 2, 4])])
        add("multi_index_broadcast", torch.randn(3, 4, 5, generator=rng), [torch.tensor([0, 2]), torch.tensor([1])])
        add("scalar_0dim", torch.randn(4, 5, generator=rng), [torch.tensor(2)])
        add("bool_mask", torch.randn(4, 5, generator=rng), [torch.tensor([True, False, True, False])])
        add("2d_mask", torch.randn(3, 4, generator=rng), [torch.randn(3, 4, generator=rng) > 0])
        add("transpose_view", torch.randn(4, 5, generator=rng).t(), [torch.tensor([1, 3])])
        add("strided_view", torch.arange(20, dtype=torch.float32)[::2], [torch.tensor([0, 4])])
        add("int32_index", torch.randn(100, generator=rng), [torch.tensor([10, 20, 30], dtype=torch.int32)])
        add("int64_index", torch.randn(100, generator=rng), [torch.tensor([10, 20, 30], dtype=torch.int64)])
        for dt in (torch.float32, torch.float64, torch.int64, torch.int32,
                   torch.int16, torch.int8, torch.uint8, torch.bool,
                   torch.float16, torch.bfloat16):
            x = torch.arange(24, dtype=torch.float32).to(dt)
            add(f"dtype_{dt}", x, [torch.tensor([0, 5, 23])])
        add("complex64", torch.randn(4, 3, dtype=torch.complex64, generator=rng), [torch.tensor([0, 2])])
        add("nd_integer_2d", torch.arange(20, dtype=torch.float32).reshape(4, 5), [torch.tensor([[0, 1], [2, 3]])])
        add("nd_integer_3d", torch.arange(20, dtype=torch.float32).reshape(4, 5), [torch.tensor([[[1], [0]], [[2], [3]]])])
        add("nd_with_row", torch.randn(6, 4, 5, generator=rng), [None, torch.tensor([[0, 1], [2, 3]])])

    if shape_set in ("large", "all"):
        # K>=1024 → linearize + gather_index（SVE/桶化）；K<1024 → gather_direct
        add("1d_large_f32", torch.randn(5000, generator=rng), [torch.randint(0, 5000, (10000,), generator=rng)])
        add("1d_large_f64_i64idx", torch.randn(4096, dtype=torch.float64, generator=rng),
            [torch.randint(0, 4096, (4096,), generator=rng)])   # f64 → idx int64
        add("1d_large_i32", torch.randint(-1000, 1000, (4096,)), [torch.randint(0, 4096, (4096,), generator=rng)])
        add("rowcopy_A4", torch.randn(200, 4, generator=rng), [torch.randint(0, 200, (2048,), generator=rng)])
        add("rowcopy_A64", torch.randn(50, 64, generator=rng), [torch.randint(0, 50, (2048,), generator=rng)])
        add("rowcopy_A4096", torch.randn(27, 64, 64, generator=rng), [torch.randint(0, 27, (2000,), generator=rng)])
        add("bucket_small_domain", torch.randn(2048, generator=rng), [torch.randint(0, 2048, (4096,), generator=rng)])
        add("bucket_small_domain_i32", torch.randn(2048, generator=rng),
            [torch.randint(0, 2048, (4096,), generator=rng).to(torch.int32)])
        add("multi_index_2d", torch.randn(5000, 5000, generator=rng),
            [torch.randint(0, 5000, (100_000,), generator=rng), torch.randint(0, 5000, (100_000,), generator=rng)])
        add("multi_index_2d_i32", torch.randn(5000, 5000, generator=rng),
            [torch.randint(0, 5000, (100_000,), generator=rng).to(torch.int32),
             torch.randint(0, 5000, (100_000,), generator=rng).to(torch.int32)])
        add("multi_index_broadcast_scalar", torch.randn(5000, 5000, generator=rng),
            [torch.randint(0, 5000, (100_000,), generator=rng), torch.tensor([42])])
        add("x_col_i", torch.randn(64, 2048, generator=rng), [None, torch.randint(0, 2048, (100_000,), generator=rng)])
        add("mask_large", torch.randn(1_000_000, generator=rng), [torch.rand(1_000_000, generator=rng) < 0.3])
        add("scalar_A3", torch.randn(200000, 3, generator=rng), [torch.randint(0, 200000, (2_000_000,), generator=rng)])
        add("noncontig_after", torch.randn(16, 64, generator=rng).t(), [torch.randint(0, 64, (1024,), generator=rng)])

    return cases


def run_gen(args):
    os.makedirs(REF_DIR, exist_ok=True)
    cases = gen_cases(args.shape_set)
    refs = {}
    for name, x, idx in cases:
        # 参考 = true aten（本进程无覆盖：HPC_INDEX_NO_OVERRIDE=1 或未 import index_sve）。
        # 注意：传给 op 的 idx 列表里 None 是"跳过维"占位（语义 == x[:, i]），
        # 不是 Python 的 newaxis；因此直接 `x[idx]` 在 None 存在时不等价，须用
        # 显式 slice 列表构造参考。这里对含 None 的用例单独处理。
        if any(t is None for t in idx):
            slices = [slice(None) if t is None else t for t in idx]
            y = x[slices]
        else:
            y = x[idx]
        refs[name] = (x, idx, y)
        print(f"gen {name}: out_shape={tuple(y.shape) if y is not None else None}")
    with open(REF_FILE, "wb") as f:
        pickle.dump(refs, f)
    print(f"wrote {len(refs)} cases -> {REF_FILE}")


def run_check(args):
    from index_sve import index as hpc_index
    with open(REF_FILE, "rb") as f:
        refs = pickle.load(f)
    failed = 0
    for name, (x, idx, y_ref) in sorted(refs.items()):
        y = hpc_index(x, idx)
        try:
            torch.testing.assert_close(y, y_ref)
            print(f"PASS {name}: shape={tuple(y.shape)} dtype={y.dtype}")
        except Exception as e:
            failed += 1
            print(f"FAIL {name}: {e!r}")
    print(f"== {len(refs) - failed}/{len(refs)} passed ==")
    sys.exit(1 if failed else 0)


def run_both(args):
    import subprocess
    env = dict(os.environ)
    env["HPC_INDEX_NO_OVERRIDE"] = "1"
    gen = subprocess.run([sys.executable, __file__, "--mode", "gen", "--shape-set", args.shape_set],
                         env=env, capture_output=True, text=True)
    print(gen.stdout, gen.stderr, file=sys.stderr if gen.returncode else sys.stdout)
    if gen.returncode:
        sys.exit(gen.returncode)
    check = subprocess.run([sys.executable, __file__, "--mode", "check", "--shape-set", args.shape_set],
                           capture_output=True, text=True)
    print(check.stdout, file=sys.stdout)
    if check.stderr:
        print(check.stderr, file=sys.stderr)
    sys.exit(check.returncode)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("gen", "check", "both"), default="check")
    p.add_argument("--shape-set", choices=("small", "large", "all"), default="all")
    args = p.parse_args()
    if args.mode == "gen":
        run_gen(args)
    elif args.mode == "check":
        run_check(args)
    else:
        run_both(args)
