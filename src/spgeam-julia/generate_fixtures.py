from pathlib import Path
import struct,json
B=Path(__file__).resolve().parent
bits=lambda x:struct.unpack('<I',struct.pack('<f',x))[0]
inputs=[];oracle=[];cases=[]
for m,n,mode in [(1,1,0),(17,33,1),(33,17,2),(257,129,3),(1024,1024,4)]:
 dense=[];rp=[0];ci=[];values=[]
 for i in range(m):
  row=[]
  for j in range(n):
   take=mode==0 or mode==1 or(mode==2 and i%5!=0 and(i*7+j*11)%13==0)or(mode==3 and(i in[0,m//2]or j==i%n))or(mode==4 and(i*17+j*13)%101<3)
   value=(((i*3+j*5)%31)-15)/8 if take else 0.;row.append(value)
   if value!=0:values.append(bits(value));ci.append(j)
  dense.append(row);rp.append(len(values))
 # Oracle follows a dense transpose scan, independently from CSR scatter.
 bp=[0];ri=[];bv=[]
 for j,column in enumerate(zip(*dense)):
  for i,value in enumerate(column):
   if value!=0:ri.append(i);bv.append(bits(value))
  bp.append(len(bv))
 c=dict(m=m,n=n,mode=mode,nnz=len(values),io=len(inputs),oo=len(oracle));cases.append(c);inputs.extend(values+rp+ci);oracle.extend(bv+bp+ri)
(B/'input_u32.bin').write_bytes(struct.pack('<%dI'%len(inputs),*inputs));(B/'oracle_u32.bin').write_bytes(struct.pack('<%dI'%len(oracle),*oracle));(B/'cases.json').write_text(json.dumps(dict(cases=cases,scope='ActualsourceisCSRtocSCtranspose,notmatrixadditiondespitename. SorteduniqueCSR/dyadicnonzerovalues, rectangular/odd/emptyrow-column controls, full independentdense transposescan oracle.'),indent=2)+'\n');(B/'study_cases.h').write_text('struct Case{int m,n,mode,nnz,io,oo;};\nstatic const Case cases[]={'+','.join('{'+','.join(str(c[k])for k in['m','n','mode','nnz','io','oo'])+'}'for c in cases)+'};\n')
