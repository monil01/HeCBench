from pathlib import Path
import json,hashlib,copy,difflib,shutil
O=Path(__file__).resolve().parent;R=Path('/home/imo/HeCBench_julia');target=Path('/tmp/hecbench-roofline-v100-20261020/batch066-rsc-secp-source-bound');sha=lambda b:hashlib.sha256(b).hexdigest();snapshot=Path('/tmp/roofline-completion-20261021/final_v19_remaining_gap_triage/snapshot_all_noncompleted.json');X=json.loads(snapshot.read_bytes());proof=[]
collector=R/'tools/v100_roofline_collect.py';assert sha(collector.read_bytes())=='c21b1edd4c6e0924250b306e1b97ff7ff1371b8407eb92899e089dbdca364116'
def replace(x,a,b):
 if isinstance(x,str):return x.replace(a,b)
 if isinstance(x,list):return [replace(v,a,b)for v in x]
 if isinstance(x,dict):return {replace(k,a,b):replace(v,a,b)for k,v in x.items()}
 return x
for n in ['rsc','secp256k1']:
 row=next(r for r in X['rows']if r['benchmark']==n and r['model']=='cuda');a=next(a for a in row['attempts']if a['profile_id']==row['latest']);old=a['files']['plan.json']['value'];oldslot=str(Path(old['scratch_cwd']).parent);newslot=str(target/n/'cuda');new=replace(old,oldslot,newslot);changes=[];additions={}
 if n=='rsc':
  before='-DRAW_DIR="/home/imo/HeCBench_julia/.porting-state/porting_logs/rsc-julia-shared/raw/cuda"';after='-DRAW_DIR="'+newslot+'/scratch/private_capture"';assert new['build_command'].count(before)==1;new['build_command']=[after if v==before else v for v in new['build_command']];changes.append({'role':'output_only_compile_definition','before':before,'after':after,'source_proof':{'path':str(R/'src/rsc-julia/study_cuda.cu'),'sha256':old['execution_bindings'][str(R/'src/rsc-julia/study_cuda.cu')],'lines':[17,18,42,43]}})
 else:
  for fixture in ['fixture.bin','expected_bytes.bin']:
   p=R/'src/secp256k1-julia'/fixture;additions[str(p)]=sha(p.read_bytes())
  new['execution_bindings'].update(additions)
 # Only a fresh attempt prefix, explicit RSC raw-output macro, and two read-only secp fixture bindings differ.
 inv=copy.deepcopy(new)
 for p in additions:del inv['execution_bindings'][p]
 if n=='rsc':inv['build_command']=[before if v==after else v for v in inv['build_command']]
 inv=replace(inv,newslot,oldslot);assert inv==old
 checks=[]
 for p,h in new['execution_bindings'].items():
  f=Path(p);actual=sha(f.read_bytes());assert actual==h,(p,h,actual);checks.append({'path':p,'sha256':h,'bytes':f.stat().st_size,'role':'unchanged_read_only_source_library_or_fixture'})
 result=Path(new['reference_result']);assert sha(result.read_bytes())==new['reference_result_sha256']
 d=O/'prospective_plans'/n/'cuda';d.mkdir(parents=True,exist_ok=False);data=(json.dumps(new,indent=2)+'\n').encode();(d/'plan.json').write_bytes(data);(d/'original_plan.json').write_bytes((json.dumps(old,indent=2)+'\n').encode());(d/'plan.diff').write_text(''.join(difflib.unified_diff(json.dumps(old,indent=2).splitlines(True),json.dumps(new,indent=2).splitlines(True),fromfile='original_frozen_plan',tofile='prospective_fresh066_plan')))
 command=['/usr/bin/python3',str(collector),'--source-root',str(R),'--plan-root',str(target),'--benchmark',n,'--skip-qualified-selected','--timeout','180'if n=='rsc'else'900','--reserve-mib','2048','--retain-auxiliary']
 proof.append({'benchmark':n,'model':'cuda','source_parent_profile':a['profile_id'],'parent_manifest_sha256':a['manifest_sha256'],'parent_frozen_plan_sha256':a['files']['plan.json']['raw_sha256'],'parent_provenance_sha256':a['files']['profile_provenance.json']['raw_sha256'],'prospective_plan_path':str(d/'plan.json'),'prospective_plan_sha256':sha(data),'configuration_id':new['configuration_id'],'raw_config_sha256':new['raw_config_sha256'],'reference_result':str(result),'reference_result_sha256':new['reference_result_sha256'],'old_slot_prefix':oldslot,'new_slot_prefix':newslot,'inverse_json_exact_equal_original':inv==old,'explicit_changes':changes,'supplemental_readonly_bindings':additions,'current_bindings_verified':checks,'root_future_launch_argv':command,'requested_build_and_profile_seconds':180 if n=='rsc'else900,'deadline_scope':'Same timeout applies independently to build and profiling; secp original180→900 for BOTH, not a build-only CLI override. RSC original180 retained.','same_kernel_math_timer_fixture_and_five_check_contract':True,'runtime_recovery_validated':False,'staging_actual':False})
(O/'plan_source_proof.json').write_text(json.dumps({'schema_version':1,'status':'prospective_private_only_not_staged_not_executed','root_reserved_target':str(target),'collector_path':str(collector),'collector_sha256':sha(collector.read_bytes()),'snapshot_sha256':sha(snapshot.read_bytes()),'records':proof},indent=2)+'\n');print('prepared',[(p['benchmark'],p['prospective_plan_sha256'])for p in proof])
