#!/usr/bin/env python3
"""Whole-array native parity plus independent source-pattern oracle."""
import argparse
from pathlib import Path
import re
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("native_prefix")
p.add_argument("iris_prefix")
p.add_argument("--elements", type=int, default=16 * 1024 * 1024)
a = p.parse_args()
assert a.elements > 0
def artifacts(prefix):
    prefix = Path(prefix)
    found = {}
    for path in prefix.parent.glob(prefix.name + ".*.bin"):
        match = re.fullmatch(r"\.(\d+)_(\d+)\.g([01])\.bin", path.name[len(prefix.name):])
        if match:
            found[tuple(map(int, match.groups()))] = path
    return found

native_files, iris_files = artifacts(a.native_prefix), artifacts(a.iris_prefix)
assert native_files and native_files.keys() == iris_files.keys(), "missing or unmatched pair artifacts"
pairs = sorted({key[:2] for key in native_files})
assert all(first < second and all((first, second, g) in native_files for g in (0, 1)) for first, second in pairs)
for first, second, gpu in sorted(native_files):
    scale = 4 if gpu == 0 else 2
    suffix = f".{first}_{second}.g{gpu}.bin"
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
    print(f"PASS pair{first}/{second} output{gpu}: all {a.elements} outputs byte-exact native and independent oracle")
