#!/usr/bin/env python3
import array,hashlib,json,math,sys
from pathlib import Path
root=Path(sys.argv[1]).resolve();rids=sys.argv[2:];R=root.parent.parent
plan=json.loads((root/'study_plan.json').read_text());prefix=plan['input_prefix_count'];out=[];stable={};seen=set()
for mode,rid in enumerate(rids):
 rp=R/'.porting-state/portable_julia/runs'/rid/'result.json';d=json.loads(rp.read_text());expected_input=(root/f'input_mode{mode}_f64.bin').read_bytes();oracle=array.array('d');oracle.frombytes((root/'oracle_f64.bin').read_bytes())
 for model,m in d['models'].items():
  for f,sha in m['adapter_file_hashes'].items():
   p=Path(f);now=hashlib.sha256(p.read_bytes()).hexdigest();assert now==sha,(f,sha,now);stable[f]=sha
  for row in m['trials']:
   a=row['output_artifact'];p=Path(a['saved_path']);b=p.read_bytes();assert hashlib.sha256(b).hexdigest()==a['sha256'];assert len(b)==207496*8;assert b[:prefix*8]==expected_input
   v=array.array('d');v.frombytes(b);errors=[abs(x-y) for x,y in zip(v[prefix:],oracle)];assert all(math.isfinite(x) for x in v);assert all(e<=1e-8+2e-13*abs(y) for e,y in zip(errors,oracle))
   at=prefix
   for spec in plan['outputs']:
    at+=spec['count'];assert list(v[at:at+17])==[plan['guard_value']]*17;at+=17
   assert at==len(v)
   out.append({'mode':mode,'model':model,'trial':row['trial'],'path':str(p.relative_to(R)),'bytes':len(b),'sha256':a['sha256'],'input_bytes_exact':len(expected_input),'output_values':len(oracle),'all_finite':True,'all_output_guards_exact':True,'max_oracle_abs_error':max(errors)})
report={'benchmark':'multimaterial','results':rids,'artifact_count':len(out),'full_value_bytes_checked':sum(x['bytes'] for x in out),'exact_input_prefix_bytes_each':prefix*8,'all60fullarrays_checked':len(out)==60,'all_pass':True,'artifacts':out,'source_bindings_checked':stable,'source_binding_count':len(stable),'criterion':{'atol':1e-8,'rtol':2e-13,'inputs_and_guards':'exactbytes','nonfinite':'reject'},'independent_oracle_provenance':'prepare_assets.py uses independently enumerated dense(cell,material)dictionary and math.fsum; separate from compact linked/CSR traversal; no ordinary source-self term silently accepted','GPU_executions_added':0}
(root/'saved_output_audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k not in ['artifacts','source_bindings_checked']},indent=2))
