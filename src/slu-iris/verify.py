#!/usr/bin/env python3
"""Compare complete native/IRIS symbolic structure, factors, solution and ABFT."""
import json
import sys
from pathlib import Path
import numpy as np

if len(sys.argv) != 5:
    raise SystemExit('Usage: verify.py native_dir iris_dir native.log iris.log')
native, iris = map(Path, sys.argv[1:3])
meta = json.loads((native / 'metadata.json').read_text())
assert meta == json.loads((iris / 'metadata.json').read_text()), 'matrix metadata mismatch'
for name in ('sym_c_ptr.u32', 'sym_r_idx.u32', 'l_col_ptr.u32', 'csr_r_ptr.u32',
             'csr_c_idx.u32', 'csr_diag_ptr.u32', 'level_idx.i32', 'level_ptr.i32'):
    a, b = (native / name).read_bytes(), (iris / name).read_bytes()
    assert a and a == b, 'symbolic structure mismatch: ' + name
for name, count in [('lu.f32', meta['nnz']), ('solution.f32', meta['n'])]:
    a = np.fromfile(native / name, dtype=np.float32)
    b = np.fromfile(iris / name, dtype=np.float32)
    assert a.size == b.size == count, 'incomplete numeric artifact: ' + name
    assert np.isfinite(a).all() and np.isfinite(b).all(), 'nonfinite result: ' + name
    error = np.abs(a.astype(np.float64) - b.astype(np.float64))
    assert np.all(error <= 1e-6 + 1e-5 * np.abs(a)), 'numeric mismatch: ' + name
    print(f'{name}: count={count} max_abs={error.max():.9g} PASS')
for filename in sys.argv[3:]:
    text = Path(filename).read_text()
    assert 'Results passed ABFT check.' in text, 'ABFT did not pass: ' + filename
    assert 'NaN found' not in text and 'More to come' not in text, 'native check failure'
level_ptr = np.fromfile(native / 'level_ptr.i32', dtype=np.int32)
assert len(level_ptr) == meta['levels'] + 1 and level_ptr[-1] == meta['n']
sizes = np.diff(level_ptr)
print('Level branches:', {str(k): int(v) for k, v in {
    's2': np.count_nonzero(sizes > 896),
    's4': np.count_nonzero((sizes > 448) & (sizes <= 896)),
    's32': np.count_nonzero((sizes > 16) & (sizes <= 448)),
    'onecol': np.count_nonzero(sizes <= 16)}.items()})
print(f"PASS n={meta['n']} nnz={meta['nnz']} levels={meta['levels']} full LU/solution/structure and native+IRIS ABFT")
