import time, torch, index_sve

SHAPES = [(512, 12, 1024), (2048, 12, 2048), (13824, 12, 2048),
          (55296, 19, 2048), (256000, 6, 2048)]

def bench(K, t, dom, reps=40):
    g = torch.Generator().manual_seed(0)
    x = torch.randn(dom)
    idx = torch.randint(0, dom, (K,), generator=g)
    torch.set_num_threads(t)
    for _ in range(30):
        y = x[idx]
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        y = x[idx]
        ts.append((time.perf_counter() - t0) * 1e6)
    ts.sort()
    return ts[len(ts) // 2]

for K, t, dom in SHAPES:
    runs = sorted(bench(K, t, dom) for _ in range(3))
    print("%7d %2d %6d  %.2f" % (K, t, dom, runs[1]))
