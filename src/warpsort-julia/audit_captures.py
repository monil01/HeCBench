#!/usr/bin/env python3
"""CPU-only full saved artifacts, independent sorting and exact guards/input audit."""
from pathlib import Path
import argparse,json,struct,hashlib,collections
p=argparse.ArgumentParser();p.add_argument('--result',required=True);p.add_argument('--model',required=True);p.add_argument('--receipt',required=True);a=p.parse_args();P=Path(__file__).resolve().parent;cs=json.loads((P/'cases.json').read_text());rpath=Path(a.result);r=json.loads(rpath.read_text());rows=r['models'][a.model]['trials'];tot=0
fl=lambda b:struct.unpack('<f',struct.pack('<I',b))[0]
for t in rows:
 d=t['output_artifact'];f=Path(d['saved_path']);raw=f.read_bytes();assert hashlib.sha256(raw).hexdigest()==d['sha256'];v=struct.unpack('<'+str(len(raw)//8)+'d',raw);k=0
 for c in cs:
  assert list(v[k:k+2])==[c['n'],c['mode']];k+=2
  assert list(v[k:k+c['n']])==c['bits'];k+=c['n']
 assert k==17056
 for c in cs:
  n=c['n'];bs=[int(x)for x in v[k:k+n]];k+=n
  assert collections.Counter((b if b&0x7fffffff else 0)for b in bs)==collections.Counter((b if b&0x7fffffff else 0)for b in c['bits']);assert all(fl(bs[i])>=fl(bs[i+1])for i in range(n-1))
  if c['mode']:
   ix=[int(x)for x in v[k:k+n]];k+=n;want=sorted(range(n),key=lambda i:(fl(c['bits'][i]),i),reverse=True)
   assert ix==want;assert bs==[c['bits'][i]for i in ix]
  assert list(v[k:k+192])==[0xa5a5a5a5]*192;k+=192
 assert k==len(v)==91402;tot+=258*3*256
out={'model':a.model,'result':a.result,'result_sha256':hashlib.sha256(rpath.read_bytes()).hexdigest(),'trials':len(rows),'full_supported_length_cases_per_trial':258,'actual_guard_bytes_all5':tot,'exact_actual_input_prefix_values_per_trial':17056,'complete_scientific_output_values_per_trial':24810,'artifact_values_per_trial':91402,'independent_oracle':'Python full Float32numeric descending plus complete numeric multiset with signedzero equivalence, exactlexicographic(key,index) sorting/rawbitmembership forindexed variant; all rawbits retained, noNaNtested','issues':[]};Path(a.receipt).write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
