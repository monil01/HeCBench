from pathlib import Path
import struct,json
from fractions import Fraction
B=Path(__file__).resolve().parent
f=[];ix=[];oracle=[];cases=[]
for n,mode in [(1,0),(17,1),(33,2),(257,3),(1024,4)]:
 a=[];vals=[];cols=[];rows=[];rp=[0];v=[((j*7)%15-7)/8 for j in range(n)]
 for i in range(n):
  for j in range(n):
   nz=mode==0 or mode==1 or (mode==2 and i%5!=0 and (i*7+j*11)%13==0) or(mode==3 and (i in [0,n//2] or j==i))or(mode==4 and (i*17+j*13)%101<3)
   value=((i*3+j*5)%31-15)/4 if nz else 0.
   a.append(value)
   if value!=0:vals.append(value);cols.append(j);rows.append(i)
  rp.append(len(vals))
 for i in range(n):oracle.append(float(sum((Fraction(a[i*n+j])*Fraction(v[j])for j in range(n)),Fraction(0))))
 c=dict(n=n,mode=mode,nnz=len(vals),ao=len(f),xo=len(f)+len(a),vo=len(f)+len(a)+len(v),rpo=len(ix),co=len(ix)+len(rp),ro=len(ix)+len(rp)+len(cols),oo=len(oracle)-n)
 cases.append(c);f.extend(a+v+vals);ix.extend(rp+cols+rows)
(B/'input_f32.bin').write_bytes(struct.pack('<%df'%len(f),*f));(B/'indices_u64.bin').write_bytes(struct.pack('<%dQ'%len(ix),*ix));(B/'oracle_f32.bin').write_bytes(struct.pack('<%df'%len(oracle),*oracle));(B/'cases.json').write_text(json.dumps(dict(cases=cases,oracle='Independent rational dense matvec, productsdyadic/32 andabsolutesumbelow2^24units soallpartialsumorders exact inFP32. NoarbitraryFP32matrixaccuracyclaim.'),indent=2)+'\n');(B/'study_cases.h').write_text('struct Case{int n,mode,nnz,ao,xo,vo,rpo,co,ro,oo;};\nstatic const Case cases[]={'+','.join('{'+','.join(str(c[k])for k in ['n','mode','nnz','ao','xo','vo','rpo','co','ro','oo'])+'}'for c in cases)+'};\n')
