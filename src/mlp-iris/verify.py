import sys,math
from pathlib import Path
a=Path(sys.argv[1]).read_text()
b=Path(sys.argv[2]).read_text()
x=[float(s.split()[2]) for s in a.splitlines() if s.startswith("VALUE ")]
y=[float(s.split()[2]) for s in b.splitlines() if s.startswith("VALUE ")]
assert x and len(x)==len(y),(len(x),len(y))
assert "PASS" in a and "PASS" in b and "FAIL" not in a and "FAIL" not in b
err=max(abs(u-v) for u,v in zip(x,y))
assert all(math.isfinite(u) and math.isfinite(v) and abs(u-v)<=0.01 for u,v in zip(x,y)),err
print(f"PASS native/IRIS all {len(x)} outputs, atol=0.01, max_abs={err}")
