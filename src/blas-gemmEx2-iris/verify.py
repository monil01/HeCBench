import sys,math
from pathlib import Path
a=[float(s.split()[3]) for s in Path(sys.argv[1]).read_text().splitlines() if s.startswith("NATIVE ")]
lines=Path(sys.argv[2]).read_text().splitlines()
b=[float(s.split()[3]) for s in lines if s.startswith("RAW ")]
assert a and len(a)==len(b),(len(a),len(b))
assert not any("FAIL" in s for s in lines)
worst=0
for i,(x,y) in enumerate(zip(a,b)):
 mode=i//(len(a)//7)
 tol=0 if mode==6 else 0.008*(1+abs(x))
 assert math.isfinite(x) and math.isfinite(y) and abs(x-y)<=tol,(i,x,y,tol)
 worst=max(worst,abs(x-y))
print(f"PASS all {len(a)} elements / seven modes against documented corrected-pointer native HIP; max_abs={worst}")
