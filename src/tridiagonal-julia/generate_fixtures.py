from pathlib import Path
from fractions import Fraction as F
from array import array
import json,sys
B=Path(__file__).resolve().parent;assert sys.byteorder=='little'
inputs=array('f');oracle=array('d');cases=[];states=[];POISON=-12345.5
def solve(a,b,c,d):
 n=len(b);cp=[];dp=[]
 for i in range(n):
  q=b[i]-(a[i]*cp[-1]if i else 0);assert q;cp.append(c[i]/q);dp.append((d[i]-(a[i]*dp[-1]if i else 0))/q)
 x=dp.copy()
 for i in range(n-2,-1,-1):x[i]-=cp[i]*x[i+1]
 for i in range(n):assert a[i]*(x[i-1]if i else 0)+b[i]*x[i]+c[i]*(x[i+1]if i<n-1 else 0)==d[i]
 return x,[-v for v in cp[:-1]]
def trans(v,n,m):return [v[j*n+i]for i in range(n)for j in range(m)]
for n,m,source in [(16,16,0),(32,32,0),(64,64,0),(128,256,0),(128,16,1)]:
 aa=[];bb=[];cc=[];dd=[];xx=[];ww=[]
 for s in range(m):
  if source:a=list(map(F,range(n)));b=list(map(F,range(1,n+1)));c=b.copy();c[-1]=F(0);rhs=b.copy()
  else:
   a=[F(0)if i==0 else F(1 if (i+s)%2 else-1,8)for i in range(n)];b=[F(2 if i%3 else 4)for i in range(n)];c=[F(0)if i==n-1 else F(1 if(i*3+s)%2 else-1,16)for i in range(n)];x=[F((i*5+s*7)%17-8,8)for i in range(n)];rhs=[a[i]*(x[i-1]if i else 0)+b[i]*x[i]+c[i]*(x[i+1]if i<n-1 else 0)for i in range(n)]
  sol,w=solve(a,b,c,rhs)
  if not source:assert sol==x
  for out,vals in [(aa,a),(bb,b),(cc,c),(dd,rhs),(xx,sol)]:out.extend(float(z)for z in vals)
  ww.append([POISON]+[float(z)for z in w])
 data=[aa,bb,cc,dd];case=dict(n=n,m=m,source_initializer=bool(source),io=len(inputs),case_id=len(cases));cases.append(case)
 for a in data:inputs.extend(a)
 wp=[ww[s][k]for k in range(n)for s in range(m)]
 for mode in range(9):
  reordered=mode in[5,7,8];post=[a.copy()for a in data]+[[POISON]*(n*m)for _ in range(3)]
  if reordered:post=[trans(bb,n,m),trans(cc,n,m),trans(dd,n,m),xx,trans(xx,n,m),trans(aa,n,m),wp if mode in[5,8]else[POISON]*(n*m)]
  else:post[4]=xx;post[6]=wp if mode==4 else[POISON]*(n*m)
  st=dict(n=n,m=m,mode=mode,io=case['io'],oo=len(oracle),case_id=case['case_id']);states.append(st)
  for a in post:oracle.extend(a)
(B/'input_f32.bin').write_bytes(inputs.tobytes());(B/'oracle_f64.bin').write_bytes(oracle.tobytes());(B/'cases.json').write_text(json.dumps(dict(cases=cases,states=states,atol=2e-5,rtol=2e-4,scope='Nine original distinct solvers/mappings overfourstrictdominantdyadiccontrols and originalchoice0initializeratN128M16. FractionThomas/fullsystemresidual/wcoefficientoracles. SourcefullM524288threeCLIs/R100unmeasured; namedR3. Reorderdimensionsmultiples16, systempowers2<=128, vec4Mmultiple4.'),indent=2)+'\n')
keys=['n','m','mode','io','oo','case_id'];(B/'study_cases.h').write_text('struct Case{int '+','.join(keys)+';};\nstatic const Case cases[]={'+','.join('{'+','.join(str(c[k])for k in keys)+'}'for c in states)+'};\n');(B/'study_cases.jl').write_text('const CASES=['+','.join('('+','.join(k+'='+str(c[k])for k in keys)+',)'for c in states)+']\n');print(len(inputs),len(oracle),len(states))
