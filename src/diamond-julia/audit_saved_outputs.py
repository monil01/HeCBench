#!/usr/bin/env python3
"""Read-only all saved full-value evidence audit, plus isolated CPU fixture regeneration."""
from pathlib import Path
import json,array,hashlib,sys,re,tempfile,subprocess,shutil
P=Path(__file__).resolve().parent;R=P.parent.parent
result=Path(sys.argv[1])if len(sys.argv)>1 else R/'.porting-state/portable_julia/runs/20261019T201718898496Z_diamond/result.json'
r=json.loads(result.read_text());expect=array.array('d');expect.frombytes((P/'oracle_primary_f64.bin').read_bytes());postref=array.array('d');postref.frombytes((P/'oracle_posterior_f64.bin').read_bytes());prefix=14039
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
records={};bindings=0;primary=0
for model,m in r['models'].items():
 assert m['status']=='native_check_pass' and m['provenance_valid'] and len(m['trials'])==5
 for path,digest in m['adapter_file_hashes'].items():
  path=Path(path);path=path if path.is_absolute()else Path(m['adapter_source_path'])/path
  assert sha(path)==digest,(path,digest);bindings+=1
 for t in m['trials']:
  a=t['output_artifact'];p=Path(a['saved_path']);assert sha(p)==a['sha256'];x=array.array('d');x.frombytes(p.read_bytes());assert x==expect;primary+=1
 # All four actual observer logs remain independently retrievable.
 logs=list((R/'.porting-state/portable_julia/runs').glob(f'*_diamond/{model}_run1.log'))
 lines=[(p,line)for p in logs for line in p.read_text().splitlines()if 'STUDY_AUXILIARY' in line and 'actual_' in line and 'path=' in line]
 # Select the source-exact current execution by path recorded at first model measurement.
 ids={'cuda':'20261019T200906306706Z_diamond','cuda_jl':'20261019T201447033429Z_diamond','ka':'20261019T201613228504Z_diamond','jacc':'20261019T201718898496Z_diamond'}
 log,line=next((p,l)for p,l in lines if ids[model]in str(p));trace=Path(line.split('path=')[1].strip());x=array.array('d');x.frombytes(trace.read_bytes());assert len(x)==27355 and x[:prefix]==expect[:prefix]
 offset=prefix;oracleoff=0;maximum=0.0
 for n in (1,128,257):
  hp=x[offset:offset+n*33+32];offset+=n*33+32;he=x[offset:offset+n+32];offset+=n+32
  assert all(v==-999 for v in hp[:16]+hp[-16:]);assert all(v==-12345 for v in he[:16]+he[-16:]);assert all(v==0 for v in he[16:-16])
  for v,ref in zip(hp[16:-16],postref[oracleoff:oracleoff+n*33]):
   e=abs(v-ref);assert e<=3e-7+3e-7*abs(ref);maximum=max(maximum,e)
  oracleoff+=n*33
 assert offset==len(x)and oracleoff==12738
 records[model]={'log':str(log.relative_to(R)),'log_sha256':sha(log),'actual_untimed_posterior_artifact':str(trace.relative_to(R)),'sha256':sha(trace),'full_values':len(x),'posterior_values':oracleoff,'max_abs_error':maximum,'all_guards_exact':True,'source_error_flags_all_zero':True,'primary_mean_last4_seconds':m['mean_seconds']}
# Generator is run in an isolated replica of its own relative directory contract.
with tempfile.TemporaryDirectory(prefix='diamond-cpu-cold-')as tmp:
 root=Path(tmp);out=root/'src/diamond-julia';out.mkdir(parents=True);src=root/'src/diamond-sycl/src/blast';src.mkdir(parents=True)
 shutil.copy2(P/'generate_assets.py',out);shutil.copy2(R/'src/diamond-sycl/src/blast/sm_blosum62.c',src)
 subprocess.run([sys.executable,str(out/'generate_assets.py')],check=True,capture_output=True)
 cold={f.name:{'sha256':sha(f),'bytes':f.stat().st_size}for f in out.iterdir()if f.suffix=='.bin'or f.name in ('study_params.h','study_params.jl','study_plan.json')}
 for name,meta in cold.items():assert sha(P/name)==meta['sha256'],name
report={'benchmark':'diamond','result':str(result.relative_to(R)),'result_sha256':sha(result),'all20_primary_artifacts_full_exact':primary==20,'primary_artifact_values_each':len(expect),'actual_input_prefix_values_each':prefix,'bound_dependency_hash_checks':bindings,'supplemental_actual_GPU_posterior_observers':records,'cold_CPU_files_exact':cold,'no_GPU_execution_in_audit':True,'posterior_observer_not_timed_primary_intermediate':True,'original_full_alignment_quality_certified':False}
(P/'saved_output_audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'primary_artifacts':primary,'bindings':bindings,'cold_files':len(cold),'models':records},indent=2))
