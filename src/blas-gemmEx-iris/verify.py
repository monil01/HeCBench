#!/usr/bin/env python3
"""Compare every emitted native and IRIS GEMM element, not just ten samples."""
import sys, math
from pathlib import Path
hip=Path(sys.argv[1]).read_text().splitlines()
iris=Path(sys.argv[2]).read_text().splitlines()
a=[float(s.split()[3]) for s in hip if s.startswith("NATIVE ")]
b=[float(s.split()[3]) for s in iris if s.startswith("RAW ")]
assert a and len(a)==len(b),(len(a),len(b))
assert not any("FAIL" in s for s in iris)
worst=0.
for i,(x,y) in enumerate(zip(a,b)):
 assert math.isfinite(x) and math.isfinite(y),(i,x,y)
 # Storage rounding is allowed for half and bfloat16; FP64/INT32 are stricter below.
 mode=i//(len(a)//9)
 tol=1e-10*(1+abs(x)) if mode==0 else (0 if mode==8 else 0.008*(1+abs(x)))
 assert abs(x-y)<=tol,(i,mode,x,y,tol)
 worst=max(worst,abs(x-y))
print(f"PASS native/IRIS all {len(a)} elements, all nine modes; max_abs_error={worst:.17g}")
