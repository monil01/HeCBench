"""Independent scalar FP32/BF16 optimizer oracle, no NumPy or GPU run required."""
from pathlib import Path
import array,ctypes,json,math,struct,sys
rp=Path(sys.argv[1]).resolve();res=json.loads(rp.read_text());lib=ctypes.CDLL('libm.so.6');lib.fmaf.argtypes=[ctypes.c_float]*3;lib.fmaf.restype=ctypes.c_float
f=lambda x:struct.unpack('<f',struct.pack('<f',x))[0]
def bf(x):
 u=struct.unpack('<I',struct.pack('<f',x))[0];u=(u+0x7fff+((u>>16)&1))&0xffff0000;return struct.unpack('<f',struct.pack('<I',u))[0]
def mm(A,B):
 m=len(A);k=len(B);n=len(B[0]);C=[]
 for i in range(m):
  row=[]
  for j in range(n):
   v=0.
   for z in range(k):v=lib.fmaf(A[i][z],B[z][j],v)
   row.append(bf(v))
  C.append(row)
 return C
def transpose(A):return [list(row)for row in zip(*A)]
def flatten(A):return [v for row in A for v in row]
shapes=[(16,32,False),(32,16,False),(17,33,True),(33,17,True)];cases=[];prefix=[]
for P,Q,dense in shapes:
 n=P*Q;p=[f((i*11%37-18)/64)for i in range(n)];m=[f((i*13%29-14)/64)if dense else f((i%Q+1)/64)if i//Q==i%Q else 0. for i in range(n)];g=[f((i*7%31-15)/128)if dense else f((i%Q+1)/128)if i//Q==i%Q else 0. for i in range(n)];scale=f(math.sqrt(max(1.,f(P/Q))));prefix.extend(p+m+g+[P,Q,2,f(.95),f(.02),0.,scale,int(dense)]);initial=p.copy();beta=f(.95);lr=f(.02)
 for step in range(2):
  update=[]
  for i in range(n):t=f(f(1.-beta)*g[i]);m[i]=lib.fmaf(beta,m[i],t);update.append(bf(lib.fmaf(beta,m[i],t)))
  X=[update[i*Q:(i+1)*Q]for i in range(P)];X=transpose(X)if P>Q else X
  # Independent high-precision norm accumulation, rounded at declared FP32 reciprocal stage.
  ss=sum(v*v for v in flatten(X));inv=f(1./f(f(math.sqrt(ss))+f(1e-7)));X=[[bf(f(v*inv))for v in row]for row in X]
  for _ in range(5):
   A=mm(X,transpose(X));A2=mm(A,A);B=[[bf(lib.fmaf(f(-4.775),a,f(f(2.0315)*a2)))for a,a2 in zip(row,row2)]for row,row2 in zip(A,A2)];BX=mm(B,X);X=[[bf(lib.fmaf(f(3.4445),v,bx))for v,bx in zip(row,br)]for row,br in zip(X,BX)]
  update=flatten(transpose(X)if P>Q else X)
  for i in range(n):p[i]=lib.fmaf(-lr,f(scale*update[i]),p[i])
 cases.append({'P':P,'Q':Q,'dense':dense,'p':p,'m':m,'X':flatten(X),'A':flatten(A),'A2':flatten(A2),'B':flatten(B),'BX':flatten(BX),'ss':ss,'scale':scale,'diagonal_singular_values_after5':[X[i][i]for i in range(min(P,Q))]if not dense else None})
report={'oracle':'Independent scalar Python FP32/FMA and explicit BF16 round-to-nearest-even matrix recurrences, high-precision norm accumulation. No native kernels or copied golden outputs.','input_prefix_count':len(prefix),'state_alignment':{'p_m':'After optimizer step2','X':'After Newton-Schulz iteration5 in optimizer step2','A_A2_B_BX':'Scratch from final Newton-Schulz iteration, formed from X at iteration4, before final X update','ss':'Squared Frobenius norm before normalization in optimizer step2'},'cases':[{'P':c['P'],'Q':c['Q'],'dense':c['dense'],'scale':c['scale'],'diagonal_singular_values_after5':c['diagonal_singular_values_after5']}for c in cases],'trials':[],'scope':'Original GPU FP32 spectral-scaling contract. CPU companion rounds scaled update toBF16 additionally, a separate arithmetic approximation. Five-step polynomial is not exact polar orthogonality or training convergence. Host norm accumulation differs from GPU reduction, so fullstate arithmetic diagnostics separate from1e-2 final-parameter acceptance.'}
for model,spec in res['models'].items():
 for tr in spec.get('trials',[]):
  data=array.array('d');data.frombytes(Path(tr['output_artifact']['saved_path']).read_bytes())
  if sys.byteorder!='little':data.byteswap()
  assert list(data[:6470])==prefix,(model,'prefix');out=data[6470:];offset=0;checks=[]
  for c in cases:
   n=c['P']*c['Q'];mmn=min(c['P'],c['Q'])**2;counts=[n,n,n,mmn,mmn,mmn,n,1];errors={}
   for name,count in zip(['p','m','X','A','A2','B','BX','ss'],counts):
    got=out[offset:offset+count];offset+=count;want=[c['ss']]if name=='ss'else c[name];errors[name]=max(abs(a-b)for a,b in zip(got,want));assert all(math.isfinite(x)for x in got)
   assert errors['p']<1e-2,(model,c['P'],c['Q'],'parameter',errors['p']);assert errors['m']<1e-6,(model,'momentum',errors['m']);checks.append({'P':c['P'],'Q':c['Q'],'max_abs_error_by_state_vs_independent_mixed_precision_oracle':errors,'parameter_acceptance':1e-2,'momentum_acceptance':1e-6})
  assert offset==len(out)==11858
  report['trials'].append({'model':model,'trial':tr['trial'],'exact_prefix':True,'checks':checks})
q=rp.parent/'scientific_optimizer_witness.json';q.write_text(json.dumps(report,indent=2)+'\n');print(q)
