import sys,math,re
from pathlib import Path
def values(p):
 return [tuple(map(float,s.split()[3:])) for s in re.findall(r"VALUE [^\n]+",Path(p).read_text())]
a,b=values(sys.argv[1]),values(sys.argv[2])
assert a and len(a)==len(b),(len(a),len(b))
worst=[0.,0.]
for i,(x,y) in enumerate(zip(a,b)):
 for j,(u,v) in enumerate(zip(x,y)):
  # LUT half2 atomic accumulation order may vary between launches.
  tol=2*(0.0625+0.01*max(abs(u),abs(v))) if j==0 else 0
  assert math.isfinite(u) and math.isfinite(v) and abs(u-v)<=tol,(i,j,u,v,tol)
  worst[j]=max(worst[j],abs(u-v))
 if len(y)>2:
  oracle=y[2]
  tol=y[3]
  assert abs(x[0]-oracle)<=tol and abs(y[0]-oracle)<=tol,(i,x[0],y[0],oracle,tol)
print(f"PASS full LUT and hipBLAS outputs: {2*len(a)} elements; max_abs LUT={worst[0]}, BLAS={worst[1]}")
