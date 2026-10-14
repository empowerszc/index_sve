import sys, traceback

def main(glob):
    tests = [(n, f) for n, f in sorted(glob.items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for n, f in tests:
        try:
            f()
            print(f"PASS {n}")
        except Exception as e:
            failed += 1
            print(f"FAIL {n}: {e!r}")
            traceback.print_exc()
    print(f"== {len(tests) - failed}/{len(tests)} passed ==")
    sys.exit(1 if failed else 0)
