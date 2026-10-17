#!/usr/bin/env python3
"""Native source table export plus independent GF(2) bit-matrix/normal oracle."""
from pathlib import Path
import subprocess,struct,statistics,math,hashlib,json,tempfile
R=Path(__file__).resolve().parent
with tempfile.TemporaryDirectory() as d:
 exe=Path(d)/'export';subprocess.run(['g++','-O2','-std=c++17',str(R/'generate_study_fixture.cpp'),'-o',str(exe)],check=True);subprocess.run([str(exe)],cwd=R,check=True)
p=list(struct.unpack('<112d',(R/'study_input_f64.bin').read_bytes()));n,seed,total,distance,repeats=map(int,p[:5]);table=list(map(int,p[5:98]));probes=list(map(int,p[98:]));U=2**32-1
# Independent output-bit rows rather than word-wise XOR recurrence.
gold=[];ranks=[];counts=[]
def rank(rows):
 basis={}
 for x in rows:
  while x:
   b=x.bit_length()-1
   if b in basis:x^=basis[b]
   else:basis[b]=x;break
 return len(basis)
for dim in range(3):
 rows=[sum(((table[dim*31+b]>>ob)&1)<<b for b in range(31)) for ob in range(31)]
 ranks.append(rank(rows));bins=[0]*16
 for pos in range(n):
  v=sum((((pos+seed)&row).bit_count()%2)<<ob for ob,row in enumerate(rows));q=struct.unpack('<f',struct.pack('<f',v+1))[0]*2**-31;gold.append(q)
  if pos<65536:bins[min(15,int(q*16))]+=1
 counts.append(bins)
normal=statistics.NormalDist()
def quantile(x):
 lo=min(x,U-x);z=normal.inv_cdf((lo+.5)/U);return -z if x>=2**31 else z
gold.extend(quantile((pos+1)*distance) for pos in range(total));gold.extend(map(quantile,probes));(R/'study_math_expected_f64.bin').write_bytes(struct.pack('<%dd'%len(gold),*gold))
assert ranks==[31]*3 and counts==[[4096]*16]*3
w={'benchmark':'qrg','algorithm':'Niederreiter GF(2) digital sequence; separate Moro inverse normal grid','N':n,'seed':seed,'stage_outputs':total,'probe_count':len(probes),'input_prefix_count':112,'artifact_count':112+len(gold),'gf2_31bit_ranks':ranks,'first_65536_points_16_bin_counts':counts,'independent_oracle':'output-bit GF(2) matrix parity; statistics.NormalDist inverse (reflected half-step probabilities)','scope':'Finite-prefix balance and pointwise transform accuracy; ICND does not consume QRNG output. Not integration/discrepancy/convergence certification.','generation_command':'python3 src/qrg-julia/generate_study_fixture.py','native_generator_flags':['g++','-O2','-std=c++17'],'files':{}}
for f in ['study_input_f64.bin','study_cjn_i64.bin','study_math_expected_f64.bin','generate_study_fixture.cpp','../qrg-cuda/reference.cu','../qrg-cuda/qrg.h']:
 b=(R/f).read_bytes();w['files'][f]={'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
(R/'scientific_operator_witness.json').write_text(json.dumps(w,indent=2)+'\n');print(json.dumps(w,indent=2))
