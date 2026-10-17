#!/usr/bin/env python3
"""Reproduce canonical Float32 subject, true statistics and radius8 query envelope."""
import struct,math,hashlib,json,argparse
from pathlib import Path
N=262144;M=256;RADIUS=8;ROOT=Path(__file__).resolve().parent
def f32(x):return struct.unpack('<f',struct.pack('<f',x))[0]
def fixture():
 subject=[f32((((i*37+11)%1021)-510)/256)for i in range(N)];s=[0.];ss=[0.]
 for x in subject:s.append(s[-1]+x);ss.append(ss[-1]+x*x)
 mean=[];std=[]
 for i in range(N-M+1):
  a=(s[i+M]-s[i])/M;v=(ss[i+M]-ss[i])/M-a*a;assert v>0;mean.append(f32(a));std.append(f32(math.sqrt(v)))
 query=[f32(f32(x-mean[0])/std[0])for x in subject[:M]]
 lower=[min(query[max(0,i-RADIUS):min(M,i+RADIUS+1)])for i in range(M)];upper=[max(query[max(0,i-RADIUS):min(M,i+RADIUS+1)])for i in range(M)]
 values=subject+mean+std+lower+upper+query;data=struct.pack('<'+'f'*len(values),*values)
 return data,{'n':N,'query_length':M,'radius':RADIUS,'order':['subjectN','meanN-M+1','stdN-M+1','lowerM','upperM','queryM'],'std_min':min(std),'std_max':max(std),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'statistics':'Populationmean/stdofactualwindows, computedFloat64fromexactdyadicprefixsums, roundedFloat32;querynormalizedwithFloat32operations','source_original_independent_random_statistics_and_unordered_bounds_unmeasured':True}
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--create',action='store_true');a=ap.parse_args();data,meta=fixture();p=ROOT/'study_input_f32.bin'
 if a.create and not p.exists():p.write_bytes(data)
 else:assert p.read_bytes()==data
 print(json.dumps(meta,indent=2))
