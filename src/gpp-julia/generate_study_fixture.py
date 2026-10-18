from pathlib import Path
import json,struct,math,hashlib
from decimal import Decimal,localcontext
p=Path(__file__).resolve().parent
specs=[(3,2,31,3,2,0),(5,3,67,5,3,1),(7,4,129,2,2,1),(2,1,1,2,1,2)]
prefix=[len(specs)];gold=[];cases=[];guard=-12345.0
D=lambda x:Decimal.from_float(float(x))
def mul(a,b):return(a[0]*b[0]-a[1]*b[1],a[0]*b[1]+a[1]*b[0])
with localcontext() as ctx:
 ctx.prec=80
 for ci,(nb,ng,nc,gx,gy,mode) in enumerate(specs):
  inv=[(g+1)*nc//ng for g in range(ng)];ind=[(i*7+3)%nc for i in range(nc)]+[nc-1] if mode==1 else list(range(nc))+[nc-1]
  wx=[3.,4.,5.]
  def z(i,tag):return(.025,.025) if mode==0 else (((i*7+tag*3)%19-9)/32,((i*5+tag*11)%23-11)/64)
  wt=[z(i,0) for i in range(ng*nc)];am=[z(i,1) for i in range(nb*nc)];an=[z(i,2) for i in range(nb*nc)];ep=[z(i,3) for i in range(ng*nc)]
  if mode==2:am=[(0.,0.) for _ in am]
  vc=[i*.025 for i in range(nc)] if mode==0 else [(i%13+1)/16 for i in range(nc)]
  arrays=[inv,ind,wx]+[[z[k] for z in arr] for arr in [wt,am,an,ep] for k in [0,1]]+[vc]
  meta=[nb,ng,nc,gx,gy];prefix+=meta
  for a in arrays:prefix+=a
  sums=[(Decimal(0),Decimal(0)) for _ in wx];abss=[0.,0.,0.];mind=math.inf
  for n in range(nb):
   for g in range(ng):
    igp=ind[inv[g]];a=am[n*nc+igp];b=an[n*nc+igp];sch=mul((D(a[0]),-D(a[1])),tuple(map(D,b)));sch=tuple(x*D(.5)*D(vc[igp]) for x in sch)
    for ig in range(nc):
     w=tuple(map(D,wt[g*nc+ig]));e=tuple(map(D,ep[g*nc+ig]))
     for iw,f in enumerate(wx):
      wd=(D(f)-w[0],-w[1]);den=wd[0]**2+wd[1]**2;mind=min(mind,float(den));v=mul(w,(wd[0],-wd[1]));v=tuple(x/den for x in v);v=mul(mul(v,e),sch);sums[iw]=tuple(sums[iw][k]+v[k] for k in [0,1]);abss[iw]+=math.hypot(*map(float,v))
  out=[float(x[0]) for x in sums]+[float(x[1]) for x in sums]
  for a in [out[:3],out[3:]]:gold +=[guard]*8+a+[guard]*8
  for a in arrays:gold +=[guard]*8+list(a)+[guard]*8
  cases.append(dict(case=ci,nb=nb,ng=ng,nc=nc,grid=[gx,gy],threads=32,mode=['source_uniform','signed_varied','exact_zero'][mode],input_values=sum(map(len,arrays)),output_complex=[[float(x) for x in z] for z in sums],min_denominator=mind,term_abs_sum=abss,complex_sum_condition=[abss[i]/max(math.hypot(*map(float,sums[i])),1e-300) if abss[i] else 0 for i in range(3)],sentinel_used=inv[-1]==nc))
write=lambda f,a:(p/f).write_bytes(struct.pack('<'+'d'*len(a),*a))
write('fixture.bin',prefix);write('oracle.bin',gold)
meta=dict(cases=cases,input_prefix_count=len(prefix),oracle_values=len(gold),artifact_values=len(prefix)+len(gold),oracle='Independent Decimal80 complex scalar arithmetic, allband/g/plane-wave terms and3frequencies; not native threaded accumulation order',atol=1e-11,rtol=1e-11,guards='Every input component/index and6output components guarded8eachside, exact; native AoS complex buffers mapped to same componentguard ordering as Julia SoA')
(p/'fixture_description.json').write_text(json.dumps(meta,indent=2)+'\n')
print(json.dumps(meta))
