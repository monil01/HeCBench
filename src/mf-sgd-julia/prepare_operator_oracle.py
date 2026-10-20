#!/usr/bin/env python3
"""Exact binary-rational gradient witness; no original/GPU solver call."""
from pathlib import Path
from fractions import Fraction as F
import struct,json,hashlib,math
p=Path(__file__).resolve().parent;f32=lambda x:struct.unpack('<f',struct.pack('<f',x))[0];u=2.0**-24;gamma=lambda n:n*u/(1-n*u);K=128;N=257;workers=12;lr=f32(.0125);lp=f32(.05);lq=f32(.075)
P=[F((i*17+j*7)%127-63,1024)for i in range(N)for j in range(K)];Q=[F((i*11+j*3)%113-56,1024)for i in range(N)for j in range(K)];origins=[]
for w in range(workers):
 s=(26757677*(5551212^w)+1)%(1<<31);s=(26757677*s+1)%(1<<31);origins.append(int(f32(f32(f32(float(s))/f32(float(1<<31)))*f32(N)))%N)
assert len(set(origins))==workers;rows=[];rational=[]
for row in origins:
 pp=P[row*K:(row+1)*K];qq=Q[row*K:(row+1)*K];dot=sum((x*y for x,y in zip(pp,qq)),F(0));rating=F(row%7-3,4);error=rating-dot;S=sum(abs(float(x*y))for x,y in zip(pp,qq));D=gamma(128)*S;E=D+u*(abs(float(rating))+abs(float(dot))+D)
 for which,old,other,lam in [('P',pp,qq,lp),('Q',qq,pp,lq)]:
  for j,(x,y)in enumerate(zip(old,other)):
   g=error*y-F.from_float(lam)*x;ideal=x+F.from_float(lr)*g;G=abs(float(y))*E+gamma(4)*((abs(float(error))+E)*abs(float(y))+abs(float(F.from_float(lam)*x)));delta=abs(lr)*G+gamma(3)*(abs(float(x))+abs(lr)*(abs(float(g))+G));bound=(delta*(1+2**-11)+2**-11*abs(float(ideal))+2**-25)*(1+1e-12);rows.append((0 if which=='P' else 1,row,j,float(ideal),bound));rational.append(f'{which}:{row}:{j}:{ideal.numerator}/{ideal.denominator}')
with(p/'operator_oracle.bin').open('wb')as f:
 f.write(struct.pack('<5I',0x4d464f31,N,K,workers,len(rows)));f.write(struct.pack('<12i',*origins))
 for which,row,j,ideal,bound in rows:f.write(struct.pack('<3idd',which,row,j,ideal,bound))
(p/'operator_oracle_generation.json').write_text(json.dumps({'pass':True,'status':'generatedbeforefirsttrainingGPUevaluation','origins':origins,'independent_updated_components':len(rows),'oracle_sha256':hashlib.sha256((p/'operator_oracle.bin').read_bytes()).hexdigest(),'exact_binary_rational_ideal_stream_sha256':hashlib.sha256('\n'.join(rational).encode()).hexdigest(),'bound_contract':str(p/'prospective_operator_and_property_contract.json'),'min_bound':min(x[-1]for x in rows),'max_bound':max(x[-1]for x in rows),'numeric_scope':'ExactFractionidealgradientfromHalf/Float32sourceinputs; independentlyderivedFP32dot/gradientplusFP16RNbound, notfittedobservederrors. Unselectedrowsmuststaybitexact.'},indent=2)+'\n');print('Exact-rationaloperatororacle',len(rows),'components;12disjointsourcewarps')
