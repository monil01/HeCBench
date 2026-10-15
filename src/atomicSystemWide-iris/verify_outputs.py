#!/usr/bin/env python3
"""Independent source-derived whole-ten-counter oracle, outside the GPU lock."""
import argparse
import json
import re
from pathlib import Path


def verify(path, loops, proof):
    text = Path(path).read_text()
    values = {int(k): int(v) for k, v in re.findall(r"^raw\[(\d+)\]=(-?\d+)$", text, re.M)}
    if set(values) != set(range(10)):
        raise AssertionError(f"{path}: missing/extra counters: {values}")
    n = 32768
    and_ref, or_ref, xor_ref = 255, 0, 255
    for tid in range(n):
        and_ref &= 2 * tid + 7
        or_ref |= 1 << (tid % 32)
        if loops % 2:
            xor_ref ^= tid
    or_ref = or_ref if or_ref < 2**31 else or_ref - 2**32
    exact = {0: 10 * n * loops, 2: n - 1, 3: 0, 4: 0, 5: 0,
             7: and_ref, 8: or_ref, 9: xor_ref}
    checks = {k: values[k] == expected for k, expected in exact.items()}
    checks.update({k: 0 <= values[k] < n for k in (1, 6)})
    assert all(checks.values()), (path, values, exact, checks)
    assert "returned OK" in text and "ERROR!" not in text, path
    if "iris_errors=" in text:
        assert "iris_errors=0" in text, path
    if proof:
        m = re.search(r"proof cpu_updates=(\d+) gpu_threads=(\d+) gpu_updates=(\d+) overlap=(\d+)", text)
        assert m and tuple(map(int, m.groups())) == (16384 * loops, 16384, 16384 * loops, 1), path
    return {"path": str(path), "loops": loops, "counters_checked": 10,
            "exact_checks": 8, "original_nondeterministic_range_checks": 2,
            "values": values, "producer_and_overlap_proof": proof, "pass": True}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--loops", required=True, type=int)
    p.add_argument("--proof", action="store_true")
    p.add_argument("logs", nargs="+")
    a = p.parse_args()
    reports = [verify(path, a.loops, a.proof) for path in a.logs]
    # Exch and CAS are schedule-dependent. Comparing their exact bits across
    # different executions would be stricter than the original algorithm.
    for key in (0, 2, 3, 4, 5, 7, 8, 9):
        assert len({r["values"][key] for r in reports}) == 1
    print(json.dumps({"independent_whole_output_pass": True, "runs": reports}, indent=2))


if __name__ == "__main__":
    main()
