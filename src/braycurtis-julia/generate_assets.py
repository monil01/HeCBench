"""Row-major exact-dyadic fixtures; independent fsum quotient-rule and finite-difference oracle."""
from pathlib import Path
import struct,math,json,hashlib,sys,platform
P=Path(sys.argv[1])if len(sys.argv)>1 else Path(__file__).resolve().parent;P.mkdir(parents=True,exist_ok=True);G=91.25;cases=[];prefix=[];outputs=[];ats=[];rts=[];fd=[]
sg=lambda x:(x>0)-(x<0)
def evaluate(A,B,go,b,m,n,k):
 Q=b*m*n;D=[0.]*Q;S=[0.]*Q;out=[0.]*Q;g1=[0.]*len(A);g2=[0.]*len(B)
 for z in range(b):
  for i in range(m):
   for j in range(n):
    q=(z*m+i)*n+j;D[q]=math.fsum(abs(A[(z*m+i)*k+t]-B[(z*n+j)*k+t])for t in range(k));S[q]=math.fsum(abs(A[(z*m+i)*k+t]+B[(z*n+j)*k+t])for t in range(k));assert S[q]>0;out[q]=D[q]/S[q]
  for i in range(m):
   for t in range(k):
    a=A[(z*m+i)*k+t];g1[(z*m+i)*k+t]=math.fsum(go[(z*m+i)*n+j]*(sg(a-B[(z*n+j)*k+t])*S[(z*m+i)*n+j]-D[(z*m+i)*n+j]*sg(a+B[(z*n+j)*k+t]))/(S[(z*m+i)*n+j]**2)for j in range(n))
  for j in range(n):
   for t in range(k):
    a=B[(z*n+j)*k+t];g2[(z*n+j)*k+t]=math.fsum(go[(z*m+i)*n+j]*(-sg(A[(z*m+i)*k+t]-a)*S[(z*m+i)*n+j]-D[(z*m+i)*n+j]*sg(A[(z*m+i)*k+t]+a))/(S[(z*m+i)*n+j]**2)for i in range(m))
 return out,D,S,g1,g2
for dtype,fmt in [(32,'f'),(64,'d')]:
 for b,m,n,k in [(1,3,5,7),(2,17,11,33),(1,65,67,129)]:
  ci=len(cases);na=b*m*k;nb=b*n*k;nq=b*m*n
  if m==3:A=[2+(i*17%97)/64 for i in range(na)];B=[.25+(i*13%83)/64 for i in range(nb)]
  else:A=[((i*17+3)%97-48)/32+1/256 for i in range(na)];B=[((i*13+11)%83-41)/32+3/256 for i in range(nb)]
  go=[((i*7+5)%31-15)/16 for i in range(nq)]
  # Exactdyadic values representableinF32/F64; no conversion-rounding uncertainty.
  for name,data in [('x1',A),('x2',B),('go',go)]:
   full=[G]*8+data+[G]*8;(P/f'case{ci}_{name}.bin').write_bytes(struct.pack('<'+fmt*len(full),*full))
  vals=evaluate(A,B,go,b,m,n,k)
  if m==3:
   assert all(0<=v<=1 for v in vals[0]);h=1e-6;maxfd=0
   for name,arr,expected in [('x1',A,vals[3]),('x2',B,vals[4])]:
    for q in range(len(arr)):
     old=arr[q];arr[q]=old+h;up=math.fsum(a*g for a,g in zip(evaluate(A,B,go,b,m,n,k)[0],go));arr[q]=old-h;dn=math.fsum(a*g for a,g in zip(evaluate(A,B,go,b,m,n,k)[0],go));arr[q]=old;maxfd=max(maxfd,abs((up-dn)/(2*h)-expected[q]))
   assert maxfd<1e-7;fd.append(dict(case=ci,epsilon=h,all_input_scalar_partials_checked=na+nb,max_absolute_difference=maxfd,budget=1e-7,method='Independent central finite differences of weighted mathematicaldistance sum vsquotient-rule analyticgradient; awayfromabs cusps.'))
  phys=[[G]*8+x+[G]*8 for x in [A,B,go]];prefix += [dtype,b,m,n,k,10,b,m,n,k,10]+sum(phys,[])+sum(phys,[])
  for v in vals:
   outputs += [G]*8+v+[G]*8;at=1e-5 if dtype==32 else 1e-12;rt=1e-4 if dtype==32 else 1e-10
   ats += [0]*8+[at]*len(v)+[0]*8;rts += [0]*8+[rt]*len(v)+[0]*8
  cases.append(dict(id=ci,dtype=dtype,b=b,m=m,n=n,k=k,repeats=10,x1_count=na,x2_count=nb,pairs=nq,domain='strictly_positive_noncusp'if m==3 else'signed_noncusp',minimum_denominator=min(vals[2])))
Npre=len(prefix);allv=prefix+outputs
for f,data in [('oracle_f64.bin',allv),('oracle_atol_f64.bin',[0.]*Npre+ats),('oracle_rtol_f64.bin',[0.]*Npre+rts)]: (P/f).write_bytes(struct.pack('<'+'d'*len(data),*data))
header='struct Case{int id,type,b,m,n,k,repeats,na,nb,nq;};\nstatic const Case cases[]={'+','.join('{%d,%d,%d,%d,%d,%d,10,%d,%d,%d}'%(c['id'],c['dtype'],c['b'],c['m'],c['n'],c['k'],c['x1_count'],c['x2_count'],c['pairs'])for c in cases)+'};\nconstexpr int PREFIX=%d,COUNT=%d;\n'%(Npre,len(allv));(P/'study_cases.h').write_text(header)
files={f.name:dict(bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest())for f in sorted(P.glob('*.bin'))}
(P/'study_plan.json').write_text(json.dumps(dict(cases=cases,input_prefix_count=Npre,artifact_count=len(allv),files=files,oracle='Pythonmath.fsum independentquotient-rule gradients andcentralfinite-difference fullscalarcontrol onsmallpositivecase; signedcases avoidbothabs cusps andzerodenominator.',finite_difference_controls=fd,source_scope='Exact4CUDAkernelbodies viaownedrowmajoraccessorshim; PyTorch/autogradwrapper integration unavailable/unverified. F32/F64 source-testprecisions; F16dispatch unmeasured.',timing='Sum6 cases(forwardR10mean+3kernelbackwardR10mean), separatecompletionbetweenforward/backward; allocations/copies/checks excluded.',numerical_contract='F32eachfieldatol1e-5 rtol1e-4; F64atol1e-12 rtol1e-10; exactdeviceinputprefix+guards; finiteallvalues. DualglobalF32budget withstricterfieldwiseindependentCPUchecks.',undefined_domain='Allzero denominator/some abs cusps not mathematicallysmooth; no validgradient/finiteoutputclaimthere.',generator_python=platform.python_version()),indent=2)+'\n');print('BRAY_ASSETS',Npre,len(allv),fd)
