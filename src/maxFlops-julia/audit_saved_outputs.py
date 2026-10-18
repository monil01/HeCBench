"""Read-only full saved fields audit with independent per-dtype acceptance. No GPU."""
from pathlib import Path
import argparse,json,hashlib,struct,math
ap=argparse.ArgumentParser();ap.add_argument('result');a=ap.parse_args();p=Path(__file__).resolve().parent;res=json.loads(Path(a.result).read_text());prefix=(p/'fixture.bin').read_bytes();gold=list(struct.unpack('<'+'d'*((p/'oracle.bin').stat().st_size//8),(p/'oracle.bin').read_bytes()));meta=json.loads((p/'fixture_description.json').read_text());rows=[]
for model,m in res['models'].items():
 for t in m['trials']:
  f=Path(t['output_artifact']['saved_path']);raw=f.read_bytes();assert raw[:len(prefix)]==prefix;v=struct.unpack('<'+'d'*((len(raw)-len(prefix))//8),raw[len(prefix):]);go=0;ma=0.;mr=0.;norm=0.;details=[]
  for c in meta['cases']:
   N=c['N'];got=v[go:go+N+16];ref=gold[go:go+N+16];go+=N+16;assert got[:8]==tuple(ref[:8]) and got[N+8:]==tuple(ref[N+8:]);ea=[]
   for x,r in zip(got[8:N+8],ref[8:N+8]):
    assert math.isfinite(x);e=abs(x-r);assert e==0 if c['exact'] else e<=c['atol']+c['rtol']*abs(r);ea.append(e);ma=max(ma,e);mr=max(mr,e/(abs(r)+1e-300));norm=max(norm,e/(c['atol']+c['rtol']*abs(r)))
   assert tuple(v[go:go+N+16])==tuple(gold[go:go+N+16]);go+=N+16;details.append(dict(bits=c['bits'],N=N,op=c['op'],lanes=c['lanes'],repeat=c['repeat'],max_abs_error=max(ea)))
  assert go==len(gold) and t['correct'];rows.append(dict(model=model,trial=t['trial'],sha256=hashlib.sha256(raw).hexdigest(),exact_input_prefix_bytes=True,all_actual_preexecution_input_and_guard_values_exact=True,max_abs_oracle_error=ma,max_rel_oracle_error=mr,max_normalized_dtype_budget=norm,path_errors=details))
assert len(rows)==20
out=dict(result=str(Path(a.result).resolve()),result_sha256=hashlib.sha256(Path(a.result).read_bytes()).hexdigest(),all_pass=True,rows=rows,all_artifact_sha256_count=len({r['sha256'] for r in rows}),no_GPU_rerun=True)
(p/'output_audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k!='rows'}))
