from pathlib import Path
import json,struct,math
from decimal import Decimal,getcontext
getcontext().prec=80
p=Path(__file__).resolve().parent;D=lambda x:Decimal.from_float(float(x));zero=Decimal(0)
def dot(a,b):return sum((x*y for x,y in zip(a,b)),zero)
def tr(a):return [list(c) for c in zip(*a)]
def mm(a,b):return [[dot(row,col) for col in tr(b)] for row in a]
def mv(a,v):return [dot(row,v) for row in a]
def add(a,b):return [[x+y for x,y in zip(ar,br)] for ar,br in zip(a,b)]
def vecadd(a,b):return [x+y for x,y in zip(a,b)]
def flat(a):return [float(x) for col in tr(a) for x in col]
specs=[(1,7,3,1,1),(17,9,4,1,1),(257,4,2,0,0),(3,0,0,0,1)];prefix=[len(specs)];gold=[];cases=[];paths=[];guard=-12345.;emin=math.inf;sym=0.
for ci,(ns,no,nf,inter,conf) in enumerate(specs):
 arrays=[[] for _ in range(7)];series=[]
 for b in range(ns):
  T=[[D(.5 if i==j else (((b+i)%5-2)/64 if j==i+1 else 0)) for j in range(8)] for i in range(8)]
  L=[[D(1 if i==j else (((i*3+j+b)%7-3)/32 if j<i else 0)) for j in range(8)] for i in range(8)];LL=mm(L,tr(L));P=[[x*D(.5) for x in row] for row in LL];Q=[[(LL[i][j]/16+D(i==j))*D(.125) for j in range(8)] for i in range(8)]
  z=[D(((b+3*i)%11-5)/16) for i in range(8)];z[0]+=1;a=[D(((b*3+i)%13-6)/32) for i in range(8)];y=[D(((b*7+t*3)%31-15)/16) for t in range(no)];mu=D(((b%7)-3)/64)
  vals=[list(map(float,y)),flat(T),list(map(float,z)),flat(Q),flat(P),list(map(float,a)),[float(mu)]]
  # Initial matrices are dyadic and exactly representable; check frozenF64roundtrip.
  assert all(D(x)==orig for x,orig in zip(vals[1],[x for col in tr(T) for x in col]));assert all(D(x)==orig for x,orig in zip(vals[3],[x for col in tr(Q) for x in col]));assert all(D(x)==orig for x,orig in zip(vals[4],[x for col in tr(P) for x in col]))
  for dest,v in zip(arrays,vals):dest.extend(v)
  series.append((T,z,Q,P,a,y,mu))
 prefix += [ns,no,nf,inter,conf]
 for v in arrays:prefix+=v
 diffs=range(8) if ci<3 else [0,7]
 for nd in diffs:
  outs=[[],[],[],[],[]]
  for T,z,Q,P,a,y,mu in series:
   P=[row[:] for row in P];a=a[:];z=z[:] if nd else [D(i==0) for i in range(8)];c=[zero]*8;c[nd]=mu if inter else zero;vs=[];fs=[];logs=zero;fc=[];ff=[]
   for it,obs in enumerate(y):
    v=obs-dot(z,a);pz=mv(P,z);F=dot(z,pz);assert F>0;emin=min(emin,float(F));vs.append(float(v));fs.append(float(F))
    if it>=nd:logs+=F.ln()
    apost=[x+zz*(v/F) for x,zz in zip(a,pz)];Ppost=[[P[i][j]-pz[i]*pz[j]/F for j in range(8)] for i in range(8)]
    a=vecadd(mv(T,apost),c);P=add(mm(mm(T,Ppost),tr(T)),Q);sym=max(sym,float(max(abs(P[i][j]-P[j][i]) for i in range(8) for j in range(8))))
   for it in range(nf):
    fc.append(float(dot(z,a)));a=vecadd(mv(T,a),c)
    if conf:
     F=dot(z,mv(P,z));assert F>0;emin=min(emin,float(F));ff.append(float(F));P=add(mm(mm(T,P),tr(T)),Q)
    else:ff.append(guard)
   for dest,v in zip(outs,[vs,fs,[float(logs)],fc,ff]):dest.extend(v)
  for v in outs:gold +=[guard]*8+v+[guard]*8
  for v in arrays:gold +=[guard]*8+v+[guard]*8
  paths.append(dict(base_case=ci,n_diff=nd,output_sizes=[len(x) for x in outs],checked_output_scalars=sum(len(x) for x in outs),expected_log_zero=nd>=no,forecast_covariance_written=bool(conf and nf)))
 cases.append(dict(case=ci,series=ns,observations=no,forecast_steps=nf,intercept=bool(inter),confidence=bool(conf),array_sizes=[len(x) for x in arrays],diffs=list(diffs)))
for name,v in [('fixture.bin',prefix),('oracle.bin',gold)]:(p/name).write_bytes(struct.pack('<'+'d'*len(v),*v))
(p/'fixture_description.json').write_text(json.dumps(dict(cases=cases,paths=paths,input_prefix_count=len(prefix),oracle_values=len(gold),artifact_values=len(prefix)+len(gold),oracle='Independent Decimal80 conditionalposterior then transition matrix formulation, not copiedT*P*Ltranspose loops; logsumskipfirstn_diff. AllinitialdyadicmatricesF64roundtripcheckedexact.',oracle_precision=dict(dtype='Decimal',decimal_digits=80),min_innovation_variance=emin,covariance_positive_definiteness='P=.5LLtranspose withunitlowertriangularL; Q=.125I+.0078125LLtranspose, strictlypositive; no numericaleigenvalueapproximationclaim',max_oracle_covariance_asymmetry=sym,atol=1e-9,rtol=1e-9),indent=2)+'\n')
