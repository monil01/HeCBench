from pathlib import Path
from fractions import Fraction
import struct,json
B=Path(__file__).resolve().parent
bits=lambda x:struct.unpack('<I',struct.pack('<f',x))[0]
inputs=[];oracle=[];cases=[]
for m,k,n,mode in [(1,1,1,0),(17,11,23,1),(33,17,19,2),(64,33,65,3),(1024,1024,1024,4)]:
 a=[];b=[]
 for i in range(m):
  row={}
  for t in range(k):
   take=mode==0 or(mode==1 and i%5!=0 and(i*7+t*11)%7<2)or(mode==2 and(i==0 or t==i%k))or(mode==3 and i%7!=0 and(i*5+t*3)%11<3)or(mode==4 and t==(i*17)%k)
   if take:row[t]=Fraction((-1 if i%2 else 1)*((i*3+t*5)%7+1),4)
  a.append(row)
 for t in range(k):
  row={}
  for j in range(n):
   take=mode==0 or(mode==1 and j%7!=0 and(t*3+j*5)%11<3)or(mode==2 and(j==0 or j==t%n))or(mode==3 and j%11!=0 and(t*7+j*3)%13<3)or(mode==4 and j==(t*19)%n)
   if take:row[j]=Fraction((-1 if j%2 else 1)*((t*5+j*3)%7+1),8)
  b.append(row)
 crows=[];max_units=0
 for i,row in enumerate(a):
  out={};absolute={}
  for t,v in row.items():
   for j,w in b[t].items():out[j]=out.get(j,Fraction(0))+v*w;absolute[j]=absolute.get(j,Fraction(0))+abs(v*w)
  assert all(v!=0 for v in out.values());max_units=max(max_units,max((int(v*32)for v in absolute.values()),default=0));crows.append(out)
 assert max_units<2**24
 # Independent dense-index dot check for compact cases.
 if mode!=4:
  for i in range(m):
   for j in range(n):assert sum((a[i].get(t,Fraction(0))*b[t].get(j,Fraction(0))for t in range(k)),Fraction(0))==crows[i].get(j,Fraction(0))
 def csr(rows):
  vals=[];rp=[0];ci=[]
  for row in rows:
   for j,v in sorted(row.items()):ci.append(j);vals.append(bits(float(v)))
   rp.append(len(vals))
  return vals,rp,ci
 av,ar,ac=csr(a);bv,br,bc=csr(b);cv,cr,cc=csr(crows);c=dict(m=m,k=k,n=n,mode=mode,annz=len(av),bnnz=len(bv),cnnz=len(cv),io=len(inputs),oo=len(oracle),max_absolute_sum_units_1_over32=max_units);cases.append(c);inputs.extend(av+ar+ac+bv+br+bc);oracle.extend(cv+cr+cc)
(B/'input_u32.bin').write_bytes(struct.pack('<%dI'%len(inputs),*inputs));(B/'oracle_u32.bin').write_bytes(struct.pack('<%dI'%len(oracle),*oracle));(B/'cases.json').write_text(json.dumps(dict(cases=cases,scope='NativeSPMMusescusparseSpGEMM sparse×sparse/computephaseonly. Sign-per-row A/sign-per-column B preventcancellation/structuralzeros; boundedquarter/eighthinputs andFractionoracle. Source1024³/1024nnzeach preserved dimensions/count via namedpermutations; sourceRNG/full1000repeats and4096caseunmeasured.'),indent=2)+'\n');(B/'study_cases.h').write_text('struct Case{int m,k,n,mode,annz,bnnz,cnnz,io,oo;};\nstatic const Case cases[]={'+','.join('{'+','.join(str(c[k])for k in ['m','k','n','mode','annz','bnnz','cnnz','io','oo'])+'}'for c in cases)+'};\n')
