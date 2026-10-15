import sys,re,math
from pathlib import Path
def parse(path):
 t=Path(path).read_text()
 mat={(int(a),int(b)):float(c) for a,b,c in re.findall(r"MATRIX (\d+) (\d+) ([^\s]+)",t)}
 target={int(a):float(b) for a,b in re.findall(r"TARGET (\d+) ([^\s]+)",t)}
 rate=re.findall(r"Cross Validation = ([^\s]+)",t)
 return t,mat,target,rate
a,x,u,r=parse(sys.argv[1]);b,y,v,s=parse(sys.argv[2])
assert x and x.keys()==y.keys() and u and u==v and r==s,(len(x),len(y),u,v,r,s)
assert "Independent full RBF matrix" in b and "PASS" in b and "FAIL" not in b
worst=0
for k,z in x.items():
 w=y[k];assert math.isfinite(z) and math.isfinite(w) and abs(z-w)<=1e-5+1e-4*abs(z),(k,z,w)
 worst=max(worst,abs(z-w))
print(f"PASS native/IRIS whole RBF matrix {len(x)} values, max_abs={worst}; all {len(u)} cross-validation targets exact; rate={r}")
if len(sys.argv)>3:
 _,z,w,t=parse(sys.argv[3]);assert z.keys()==x.keys() and u==w and r==t
 err=max(abs(x[k]-z[k]) for k in x)
 assert all(abs(x[k]-z[k])<=1e-5+1e-4*abs(z[k]) and abs(y[k]-z[k])<=1e-5+1e-4*abs(z[k]) for k in z),err
 print(f"PASS independent CPU-only full RBF/SVM reference: {len(z)} matrix values, {len(w)} predictions exact; max_abs_native={err}")
