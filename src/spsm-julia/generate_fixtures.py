from pathlib import Path
from fractions import Fraction
from array import array
import json,sys
assert sys.byteorder=='little'
B=Path(__file__).resolve().parent
inputs=array('I');oracle=array('f');cases=[]
def raw(a):v=array('I');v.frombytes(a.tobytes());return v
for m,n,nnz,mode in [(1,1,1,0),(17,3,50,1),(33,5,150,2),(257,7,1536,3),(20082,128,150616,4)]:
 vals=array('f');rp=array('I',[0]);ci=array('I');rows=[];remaining=nnz-m
 for i in range(m):
  num=min(7,i,remaining);remaining-=num;cols=set();t=0
  while len(cols)<num and t<100:cols.add((i//2+t*13)%i);t+=1
  for j in range(i):
   if len(cols)>=num:break
   cols.add(j)
  row=[]
  for j in sorted(cols):
   sign=-1 if(i+j)%2 else 1;ci.append(j);vals.append(sign/64);row.append((j,sign))
  diag=2 if i%3 else 4;ci.append(i);vals.append(diag);rp.append(len(vals));rows.append((diag,row))
 assert remaining==0 and len(vals)==nnz
 xu=array('h',(0 if mode==1 else(i*7+j*11)%15-7 for i in range(m)for j in range(n)));bu=array('i');rhs=array('f');x=array('f',(v/8 for v in xu));maxabsunits=0
 for i,(diag,row)in enumerate(rows):
  for j in range(n):
   units=diag*64*xu[i*n+j]+sum(sign*xu[k*n+j]for k,sign in row);bound=abs(diag*64*xu[i*n+j])+sum(abs(xu[k*n+j])for k,sign in row);maxabsunits=max(maxabsunits,bound);bu.append(units);rhs.append(units/512)
 assert maxabsunits<2**23
 if m<1000:
  y=[]
  for i,(diag,row)in enumerate(rows):
   line=[]
   for j in range(n):
    value=(Fraction(bu[i*n+j],512)-sum((Fraction(sign,64)*y[k][j]for k,sign in row),Fraction(0)))/diag;assert value==Fraction(xu[i*n+j],8);line.append(value)
   y.append(line)
 c=dict(m=m,n=n,nnz=nnz,mode=mode,io=len(inputs),oo=len(oracle),max_abs_row_product_units_1_over512=maxabsunits,strict_lower_abs_row_sum_upper=7/64,minimum_diagonal=2,maximum_diagonal=4);cases.append(c);inputs.extend(raw(vals));inputs.extend(rp);inputs.extend(ci);inputs.extend(raw(rhs));oracle.extend(x)
(B/'input_u32.bin').write_bytes(inputs.tobytes());(B/'oracle_f32.bin').write_bytes(oracle.tobytes());(B/'cases.json').write_text(json.dumps(dict(cases=cases,scope='Source20082rows/128RHS/150616nnzfootprint retainedwithnamedstrictlydominantlowertriangularCSR, notoriginalrandominitializer; itsconditioningunmeasured. Exactquarterpower2diag/offdiag1/64 andknownXinteger/8, B integer/512. OutputinitializedPOISON/notknownX, noGPUoracleupload. CompactFractionforwardsolve, fullknownsolution/residual audits.'),indent=2)+'\n');(B/'study_cases.h').write_text('struct Case{int m,n,nnz,mode,io,oo;};\nstatic const Case cases[]={'+','.join('{'+','.join(str(c[k])for k in['m','n','nnz','mode','io','oo'])+'}'for c in cases)+'};\n')
