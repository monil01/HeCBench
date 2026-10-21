from pathlib import Path
import json,hashlib,datetime,fcntl
D=Path(__file__).resolve().parent;R=Path('/tmp/hecbench-roofline-v100-20261020');S=Path('/home/imo/HeCBench_julia')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ref(p):
 p=Path(p);return {'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size}
slots=[];bindings={}
for batch,bench,models in [('batch043-atomic-app-replay','atomicSystemWide',['cuda','cuda_jl','ka','jacc']),('batch057-atomic-app-recovery','atomicSystemWide',['cuda_jl','jacc']),('batch040-miniDGS','miniDGS',['cuda','cuda_jl','ka','jacc']),('batch023-completion-a','miniDGS',['cuda','cuda_jl','ka','jacc'])]:
 for m in models:
  b=R/batch/bench/m;p=b/'plan.json';v=json.loads(p.read_bytes());rec={'benchmark':bench,'model':m,'batch':batch,'plan':ref(p),'configuration_id':v['configuration_id'],'selected_contract_fields':{k:v.get(k)for k in ['config_sha256','result_sha256','raw_config_sha256','selected_result_sha256','source_hash','profile_target_command','build_command','build_commands','environment','execution_bindings','application_replay_review','conditional_replay_review','completion_overlay_provenance']}}
  for name in ['profile_provenance.json','counters.csv','execution.log','application_replay_pass_artifacts.json']:
   q=b/name
   if q.exists():
    rec[name]=ref(q)
    if name=='profile_provenance.json':
     z=json.loads(q.read_bytes());rec['observed_status']={k:z.get(k)for k in ['status','profile_process','build_process','stop_reason','error']}
  if 'observed_status'not in rec:rec['observed_status']='no_saved_profile_provenance_at_snapshot'
  slots.append(rec)
files=[S/'src/miniDGS-julia/study_driver.py',S/'src/miniDGS-julia/libstudy_mpi.so',Path('/tmp/roofline-completion-20261021/uncovered_custom_followups_056_058/launch_review.json'),Path('/tmp/roofline-completion-20261021/uncovered_custom_followups_056_058/future_controller_receipt.json'),Path('/tmp/roofline-completion-20261021/profiler2025_method_patch/v5/proposed/tools/v100_roofline_collect.py'),Path('/tmp/roofline-completion-20261021/profiler2025_method_patch/v5/proposed/tools/v100_profiler_method.py'),Path('/tmp/roofline-completion-20261021/profiler2025_calibration003/source_binding_manifest.json'),Path('/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/profilers/Nsight_Compute/docs/NsightComputeCli/index.html'),R/'batch040-miniDGS/process_filter_selector.json']
for p in files:bindings[str(p)]=ref(p)
with open('/tmp/hecbench-roofline-metadata.lock','r')as lock:
 fcntl.flock(lock,fcntl.LOCK_SH)
 selected={b:json.loads((S/f'ontology/studies/v100_roofline_537/benchmarks/{b}.json').read_bytes())for b in ['atomicSystemWide','miniDGS']}
 selected_refs={b:ref(S/f'ontology/studies/v100_roofline_537/benchmarks/{b}.json')for b in selected}
 fcntl.flock(lock,fcntl.LOCK_UN)
(D/'selected_snapshot.json').write_text(json.dumps(selected,indent=2)+'\n')
review={'schema_version':1,'created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'PROSPECTIVE_AUTHORING_SPEC_ONLY_NOT_READY_FOR_EXECUTION','method_adopted':False,'calibration003_outcome_observed_by_this_review':False,'gpu_runs':0,'builds':0,'tests':0,'shared_mutations':0,'slots':slots,'bindings':bindings,'selected_snapshot_bindings':selected_refs,'snapshot_note':'Selected documents read under short shared metadata lock; raw plans/provenance read sequentially. Mutable future-controller state is only a time-stamped observation. Root must revalidate at adoption.'}
(D/'review.json').write_text(json.dumps(review,indent=2)+'\n')
