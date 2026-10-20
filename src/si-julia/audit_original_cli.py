#!/usr/bin/env python3
from pathlib import Path
import argparse,struct,re,json,math,hashlib
p=argparse.ArgumentParser();p.add_argument('dataset',type=Path);p.add_argument('counts',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();raw=a.dataset.read_bytes();k,u,total=struct.unpack_from('<IIQ',raw);sizes=struct.unpack_from('<'+'I'*k,raw,16);data=struct.unpack_from('<'+'I'*total,raw,16+4*k);sets=[];off=0
for n in sizes:sets.append(set(data[off:off+n]));off+=n
seen=set();wrong=[];maximum=0.;nonfinite=0
for line in a.counts.read_text().splitlines():
 m=re.fullmatch(r'\((\d+),(\d+)\): (.+)',line);assert m,line;i,j=int(m[1])-1,int(m[2])-1;assert 0<=i<j<k and(i,j)not in seen;seen.add((i,j));actual=float(m[3]);expected=len(sets[i]&sets[j]);finite=math.isfinite(actual);nonfinite+=not finite
 if actual!=expected:
  if finite:maximum=max(maximum,abs(actual-expected))
  wrong.append({'i':i,'j':j,'observed':actual if finite else None,'raw_text':m[3],'expected':expected})
assert len(seen)==k*(k-1)//2
r={'status':'pass'if not wrong else'numerical_rejection','dataset':str(a.dataset),'dataset_sha256':hashlib.sha256(raw).hexdigest(),'counts':str(a.counts),'counts_sha256':hashlib.sha256(a.counts.read_bytes()).hexdigest(),'checked_all_distinct_pairs':len(seen),'wrong_pairs':len(wrong),'nonfinite_pairs':nonfinite,'max_abs_error':maximum,'wrong_pair_details':wrong,'scope':'FullactualCLIserializedpaircounts, notrawGPUarrays/unusedpadding; independentPythonsetintersectioneachpair'};a.output.write_text(json.dumps(r,indent=2)+'\n');print({k:v for k,v in r.items()if k!='wrong_pair_details'})
