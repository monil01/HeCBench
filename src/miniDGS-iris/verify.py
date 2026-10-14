#!/usr/bin/env python3
"""Compare all six nodal fields to a separate CUDA-derived CPU reference."""
import sys
from pathlib import Path
import numpy as np

if len(sys.argv) != 3:
    raise SystemExit("Usage: verify.py reference_fields iris_fields")
reference, actual = map(Path, sys.argv[1:])
ref_files = sorted(reference.glob("rank_*.bin"))
iris_files = sorted(actual.glob("rank_*.bin"))
if len(ref_files) != 2 or [p.name for p in ref_files] != [p.name for p in iris_files]:
    raise SystemExit("FAIL missing two-rank field artifacts")
names = ["Hx", "Hy", "Hz", "Ex", "Ey", "Ez"]
count = 0
passed = True
for ref_path, iris_path in zip(ref_files, iris_files):
    ref = np.fromfile(ref_path, dtype=np.float64)
    iris = np.fromfile(iris_path, dtype=np.float64)
    if not ref.size or ref.size % 6 or iris.shape != ref.shape:
        raise SystemExit("FAIL field dimensions")
    ref, iris = ref.reshape(6, -1), iris.reshape(6, -1)
    if not np.isfinite(ref).all() or not np.isfinite(iris).all():
        raise SystemExit("FAIL nonfinite fields")
    for name, expected, result in zip(names, ref, iris):
        error = np.abs(result - expected)
        good = np.all(error <= 2e-6 + 2e-5 * np.abs(expected))
        print(f"{ref_path.stem} {name}: count={error.size} max_abs={error.max():.9g} {'PASS' if good else 'FAIL'}")
        count += error.size
        passed &= good
if not passed:
    raise SystemExit(1)
print(f"PASS {count} nodal field values, atol=2e-6 rtol=2e-5; no-native-hip-baseline")
