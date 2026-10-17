#!/usr/bin/env python3
from array import array
from pathlib import Path
import math,json,hashlib
p=Path(__file__).resolve().parent;inputs=array('f');gold=array('d');T=257;E=384
for pattern in [0,1]:
 a=array('f')
 for t in range(T):
  for e in range(E):
   x=((37*e+17*t)%401-200)/32 if pattern==0 else 0 if t%3==0 else (-100000+(e%5)*50000) if t%3==1 else (100000 if e==E-1 else -100000)
   a.append(x)
 inputs.extend(a);sm=[];orders=[]
 for t in range(T):
  row=a[t*E:(t+1)*E];m=max(row);w=[math.exp(x-m)for x in row];z=math.fsum(w);w=[x/z for x in w];sm.extend(w);orders.append(sorted(range(E),key=lambda e:(-w[e],e)))
 for k in [1,2,4,8]:
  gold.extend(sm);gold.extend(sm[t*E+orders[t][j]]for t in range(T)for j in range(k));gold.extend(orders[t][j]for t in range(T)for j in range(k));gold.extend(j*T+t for t in range(T)for j in range(k))
f=p/'moe_input_float32.bin';f.write_bytes(inputs.tobytes());g=p/'moe_expected_float64.bin';g.write_bytes(gold.tobytes());d={'tokens':T,'experts':E,'k':[1,2,4,8],'patterns':['boundeddyadic','large/tied/source-scale'],'input_count':len(inputs),'output_count':len(gold),'oracle':'IndependentPythonFloat64 naturalexp+math.fsum normalization andstablefullsort (-probability,expertid), notiterativeGPUreduction.','hashes':{x.name:hashlib.sha256(x.read_bytes()).hexdigest()for x in [f,g]},'max_integer_output':8*T-1,'scope':'Eightnamedroutingcases, nofinished/expertpartitionparameters orfullmodelclaim'};(p/'fixture_provenance.json').write_text(json.dumps(d,indent=2)+'\n');print(d)
