from pathlib import Path
from array import array
from fractions import Fraction
import json,sys,hashlib
B=Path(__file__).resolve().parent;W=B.parents[1]/'.porting-state/portable_julia/sptrsv_reference_review'
assert sys.byteorder=='little'
def load(n,t):a=array(t);a.frombytes((W/n).read_bytes());return a
inputs=array('d');oracle=array('d');cases=[]
def grouping(m,rp):
 mp=[0]
 for i in range(0,m,32):
  end=min(m,i+32)
  if (rp[end]-rp[i])/(end-i)>=10:mp.extend(range(i+1,end+1))
  else:mp.append(end)
 return mp
def add(name,rp,ci,v,b,x,mp):
 m=len(b);assert len(x)==m and len(rp)==m+1 and len(ci)==len(v)==rp[-1] and list(mp)==grouping(m,rp)
 maxbound=0;depth=[0]*m
 for i in range(m):
  row=ci[rp[i]:rp[i+1]];assert row[-1]==i and all(a<b for a,b in zip(row,row[1:]))and all(v[j]==1 for j in range(rp[i],rp[i+1]))
  total=sum(x[k]for k in row);assert total==b[i];maxbound=max(maxbound,sum(abs(x[k])for k in row));depth[i]=1+max((depth[k]for k in row[:-1]),default=0)
 if m<1000:
  y=[]
  for i in range(m):
   z=Fraction(b[i])-sum((y[k]for k in ci[rp[i]:rp[i+1]-1]),Fraction(0));assert z==Fraction(x[i]);y.append(z)
 c=dict(name=name,m=m,nnz=len(ci),length=len(mp),io=len(inputs),oo=len(oracle),thread_groups=sum(mp[k+1]-mp[k]>1 for k in range(len(mp)-1)),warp_rows=sum(mp[k+1]-mp[k]==1 for k in range(len(mp)-1)),dependency_depth=max(depth),max_abs_row_sum=maxbound)
 cases.append(c)
 for values in [rp,ci,v,b,mp]:inputs.extend(float(v)for v in values)
 oracle.extend(x)
for name,m,kind in [('bidiagonal17',17,0),('denseLower32',32,1),('mixed65',65,2),('pattern257',257,3)]:
 rp=[0];ci=[]
 for i in range(m):
  if kind==0:cols=([i-1]if i else[])+[i]
  elif kind==1:cols=list(range(i+1))
  elif kind==2:cols=list(range(i+1))if i<32 else(([i-1]if i else[])+[i])
  else:cols=sorted(set([i]+([i-1,i//2]if i else[])+list(range(max(0,i-22),i))))
  ci.extend(cols);rp.append(len(ci))
 v=[1.]*len(ci);x=[((i*13)%17-8)/8 for i in range(m)];b=[sum(x[k]for k in ci[rp[i]:rp[i+1]])for i in range(m)];add(name,rp,ci,v,b,x,grouping(m,rp))
rp=load('lp1_rp.i32','i');ci=load('lp1_ci.i32','i');v=load('lp1_val.f64','d');b=load('lp1_b.f64','d');x=load('lp1_x.f64','d');mp=load('lp1_map.i32','i');assert all(a==1 for a in x);add('original_lp1_pattern_unit_lower',rp,ci,v,b,x,mp)
(B/'input_f64.bin').write_bytes(inputs.tobytes());(B/'oracle_f64.bin').write_bytes(oracle.tobytes());(B/'cases.json').write_text(json.dumps(dict(cases=cases,scope='OriginalLP1 sourceheadersread/symmetricexpand/change2tranvalues1+unitdiag/get_x_b ones/matrix_warp4border10. Thisisdependency-topologyunit-lowersolve NOT originalnumericoptimizationproblem. CompactvarieddyadicRHS/exactFractionforwardsolve; independentfullCSRknownsolution/residual/topology/grouping/depthchecks.'),indent=2)+'\n');(B/'study_cases.h').write_text('struct Case{int m,nnz,length,io,oo;};\nstatic const Case cases[]={'+','.join('{'+','.join(str(c[k])for k in['m','nnz','length','io','oo'])+'}'for c in cases)+'};\n');print(cases)

(B/"study_cases.jl").write_text("const CASES=["+",".join("("+",".join(k+"="+str(c[k])for k in ["m","nnz","length","io","oo"])+",)"for c in cases)+"]\n")
