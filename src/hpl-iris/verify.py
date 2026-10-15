import sys,re,numpy as np
from pathlib import Path
native=Path(sys.argv[1]);iris=Path(sys.argv[2])
logs=[Path(sys.argv[3]).read_text(),Path(sys.argv[4]).read_text()]
counts=[]
for text in logs:
 passed=re.findall(r"\|\|Ax-b\|\|[^\n]+PASSED",text)
 assert len(passed)==2 and "FAILED" not in text,("HPL residual verdict",len(passed))
 z=re.findall(r"GPU_VENDOR_CALLS dgemm=(\d+) dtrsm=(\d+)",text)
 assert z and len(z)==1 and int(z[0][0])>0 and int(z[0][1])>0,z
 counts.append(z)
assert counts[0]==counts[1],counts
a=sorted(native.glob("output.rank*.case*.bin"));b=sorted(iris.glob("output.rank*.case*.bin"))
assert len(a)==len(b)==2,(a,b)
total=0
for p,q in zip(a,b):
 h=np.fromfile(p,np.int32,count=7);i=np.fromfile(q,np.int32,count=7)
 assert np.array_equal(h,i) and h[5]==0,(h,i)
 x=np.fromfile(p,np.float64,offset=28);y=np.fromfile(q,np.float64,offset=28)
 expected=int(h[2])*int(h[3])+int(h[3])-1
 assert x.size==y.size==expected,(x.size,y.size,expected)
 assert np.isfinite(x).all() and np.isfinite(y).all()
 error=np.abs(x-y);assert np.allclose(x,y,atol=1e-10,rtol=1e-10),(error.max(),np.argmax(error))
 total+=expected
 print(f"PASS HPL N={h[0]} NB={h[1]} full {expected} active-factor/RHS/solution doubles; max_abs={error.max()}, differing={np.count_nonzero(error)}")
print(f"PASS all {total} doubles; both original full-system residual checks pass; actual vendor GPU calls native=IRIS {counts[0]}")
