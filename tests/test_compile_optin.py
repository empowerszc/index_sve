import os, sys, subprocess, tempfile, textwrap, torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
import index_sve  # noqa: F401
from index_sve import compile_support


def _run(script, env_extra, timeout=900):
    env = dict(os.environ)
    env.pop("HPC_INDEX_COMPILE", None)
    env.pop("HPC_INDEX_NO_OVERRIDE", None)
    env.update(env_extra)
    env["PYTHONPATH"] = REPO
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        capture_output=True, text=True, env=env, timeout=timeout)


def test_disabled_by_default():
    if os.environ.get("HPC_INDEX_COMPILE") == "1":
        return
    assert not compile_support.enabled()


def test_optin_fake_and_compile():
    tmp = tempfile.mkdtemp(prefix="hpc_tind_")
    r = _run("""
        import os, torch
        import index_sve
        from index_sve import compile_support
        assert compile_support.enabled()
        from torch._inductor.runtime import runtime_utils
        assert runtime_utils.cache_dir().rstrip(os.sep).endswith("hpc_index"), \\
            runtime_utils.cache_dir()

        # fake: 整数下标静态形状 + mask 数据相关形状
        from torch._subclasses.fake_tensor import FakeTensorMode
        with FakeTensorMode():
            x = torch.empty(10, 3, device="meta")
            i = torch.empty(2, 3, dtype=torch.long, device="meta")
            y = torch.ops.hpc_index.index(x, [i])
            assert tuple(y.shape) == (2, 3, 3), y.shape
            o = torch.empty(0, 3, device="meta")
            assert torch.ops.hpc_index.index_out(x, [i], o) is o
            assert tuple(o.shape) == (2, 3, 3), o.shape

        N = 200_000
        x = torch.randn(N)
        i = torch.randint(0, N, (N,))
        def f(a, idx): return a[idx]
        torch._dynamo.reset()
        c = torch.compile(f)
        assert torch.equal(c(x, i), x[i])
        with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU]) as p:
            c(x, i)
        keys = [e.key for e in p.key_averages()]
        assert any(k.startswith("aten::index") for k in keys), keys

        m = torch.rand(N) < 0.3
        def g(a, msk): return a[msk]
        torch._dynamo.reset()
        cg = torch.compile(g)
        assert torch.equal(cg(x, m), x[m])
        print("OK")
    """, {"HPC_INDEX_COMPILE": "1", "TORCHINDUCTOR_CACHE_DIR": tmp})
    assert r.returncode == 0, r.stdout + r.stderr
    assert "OK" in r.stdout, r.stdout + r.stderr
    # 缓存重定向确实落在独立子目录
    assert os.path.isdir(os.path.join(tmp, "hpc_index")), os.listdir(tmp)


def test_optin_requires_override():
    r = _run("""
        import index_sve
    """, {"HPC_INDEX_COMPILE": "1", "HPC_INDEX_NO_OVERRIDE": "1"}, timeout=300)
    assert r.returncode != 0
    assert "HPC_INDEX_NO_OVERRIDE" in r.stderr, r.stderr[-500:]


if __name__ == "__main__":
    import _harness
    _harness.main(globals())
