"""Exact independent saved-capture audit; no GPU execution."""
from pathlib import Path
import json,struct,hashlib
from generate_fixture import DIVISORS,quotient
B=Path(__file__).resolve().parent;R=B.parents[1];p=R/'.porting-state/portable_julia/runs/20261018T043538867378Z_idivide/result.json';d=json.loads(p.read_text());inputbytes=(B/'input_i32.bin').read_bytes();xs=[v[0] for v in struct.iter_unpack('<i',inputbytes)];n=len(xs);stride=2*n+3;g=-123456789;prefix=struct.pack('<i',g)+inputbytes+struct.pack('<i',g);rows=[];issues=[];hashes=set()
for model,m in d['models'].items():
 for t in m['trials']:
  a=t['output_artifact'];data=Path(a['saved_path']).read_bytes();sha=hashlib.sha256(data).hexdigest();v=[z[0] for z in struct.iter_unpack('<i',data)];ok=data[:len(prefix)]==prefix and sha==a['sha256'] and len(v)==n+2+56*stride+28*6;off=n+2;bad=0
  for j,divisor in enumerate(DIVISORS):
   qs=[quotient(x,divisor) for x in xs];rs=[x-q*divisor for x,q in zip(xs,qs)]
   for method in range(2):
    base=off+(j*2+method)*stride;got=v[base:base+stride];bad+=got!=[g,divisor,*qs,*rs,g]
  tail=v[off+56*stride:];ok &=tail==[z for _ in DIVISORS for z in [g,0,0,0,0,g]] and bad==0;row={'model':model,'trial':t['trial'],'pass':ok,'values':len(v),'artifact_sha256':sha,'exact_quotient_remainder_pairs_per_method':n*len(DIVISORS)};rows.append(row);hashes.add(sha)
  if not ok:issues.append(row)
s={'result':str(p.relative_to(R)),'result_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'checked_artifacts':len(rows),'unique_artifact_hashes':sorted(hashes),'scope':'Observable capture maps call unchanged native fastdiv helper; original no-output throughput/latency CLI not certified. Actual original functional check covers +/-0..4095 and28 divisors outside timed ROI.','oracle':'Arbitrary-precision signed abs// truncation and n-q*d; exact all input bits, passed divisor fields and guards','checks':rows,'mean_seconds':{k:v['mean_seconds'] for k,v in d['models'].items()},'issues':issues};(B/'output_audit.json').write_text(json.dumps(s,indent=2)+'\n');print(json.dumps({'artifacts':len(rows),'unique_hashes':len(hashes),'issues':issues}));assert len(rows)==20 and not issues
