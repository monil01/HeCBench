#!/usr/bin/env python3
"""Whole-array native parity plus independent source-pattern oracle."""
import argparse
from pathlib import Path
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("native_prefix")
p.add_argument("iris_prefix")
p.add_argument("--elements", type=int, default=16 * 1024 * 1024)
a = p.parse_args()
assert a.elements > 0
for gpu, scale in ((0, 4), (1, 2)):
    suffix = f".0_1.g{gpu}.bin"
    native_path, iris_path = (Path(prefix + suffix) for prefix in (a.native_prefix, a.iris_prefix))
    assert native_path.stat().st_size == iris_path.stat().st_size == a.elements * 4
    native = np.memmap(native_path, dtype=np.float32, mode="r")
    iris = np.memmap(iris_path, dtype=np.float32, mode="r")
    for start in range(0, a.elements, 1048576):
        end = min(start + 1048576, a.elements)
        expected = ((np.arange(start, end, dtype=np.int64) % 4096) * scale).astype(np.float32)
        assert np.isfinite(native[start:end]).all() and np.isfinite(iris[start:end]).all()
        assert np.array_equal(native[start:end], iris[start:end]), f"native mismatch GPU{gpu} at chunk {start}"
        assert np.array_equal(iris[start:end], expected), f"oracle mismatch GPU{gpu} at chunk {start}"
    print(f"PASS GPU{gpu}: all {a.elements} outputs byte-exact native and independent oracle")
