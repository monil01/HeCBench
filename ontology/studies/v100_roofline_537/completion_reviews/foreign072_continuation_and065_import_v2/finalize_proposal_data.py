from pathlib import Path
import json,hashlib,copy,fcntl,ast
O=Path(__file__).resolve().parent;P=O.parent/'tgvnn064_foreign065_recovery_proposal_v2';R=Path('/home/imo/HeCBench_julia');sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
# AST is source inspection only; proposed modules are not imported, executed or tested.
for name in ['controller072.py','import_nested065.py']:ast.parse((O/name).read_text())
old=json.loads((P/'launch_review.json').read_bytes());data=json.loads((O/'continuation_jobs.json').read_bytes());jobs=data['jobs']
for j in jobs:j['former065_future_plan_path']=next(k['future_plan_path']for k in old['jobs']if k['job_id']==j['job_id'])
save(O/'continuation_jobs.json',data)
packet={str(O/'continuation_jobs.json'):sha(O/'continuation_jobs.json'),str(P/'controller_foreign_receipt.json'):sha(P/'controller_foreign_receipt.json'),str(P/'controller.py'):sha(P/'controller.py'),str(P/'launch_review.json'):sha(P/'launch_review.json')}
proposal={'status':'private_pending_new_root_authorization_no_execution','source_root':str(R),'controller_sha256':sha(O/'controller072.py'),'tool_bindings':old['tool_bindings'],'packet_bindings':packet,'predecessors':old['predecessors'],'jobs':jobs,'finite_bound':26,'attempted_atomicCUDAjl_excluded':True,'TGVNN_not_included_or_released_by_this_controller':True,'raw_layout':'batch072-<job_id>/<benchmark>/<model>/profile_provenance.json, exact existing flat scanner depth','stable_GPU6_idle_gate':{'minimum_seconds':10,'minimum_samples':6,'interval_seconds':2,'under_existing_GPU_lock':True,'release_before_C21_reacquires':True,'GPU5_queries_or_permission':False,'wait_may_be_unbounded':True},'parent065_start_ticks':'not_recorded; no invented identity. Exact ended receipt bound; hold ifPID319164present/reused.','storage':'Rootexplicit new scratch manifest-before-cleanup; C21unchanged. No old evidence deletion; binarypayloadsnotretained/recompared.'};save(O/'proposal072.json',proposal)
proof=[]
with Path('/tmp/hecbench-roofline-metadata.lock').open('r')as lk:
 fcntl.flock(lk,fcntl.LOCK_SH)
 for j in jobs:
  previous=json.loads(Path(j['former065_prospective_plan']['path']).read_bytes());fresh=json.loads(Path(j['prepared_plan']['path']).read_bytes());oldroot=next(k['plan_root']for k in old['jobs']if k['job_id']==j['job_id']);newroot=j['plan_root']
  def inv(v):
   if isinstance(v,str):return v.replace(newroot,oldroot)
   if isinstance(v,list):return[inv(z)for z in v]
   if isinstance(v,dict):return{inv(k):inv(z)for k,z in v.items()}
   return v
  assert inv(fresh)==previous;assert not Path(j['former065_future_plan_path']).parent.exists();assert not Path(j['plan_root']).exists();assert Path(j['future_plan_path']).parent.parent.parent==Path(j['plan_root']);assert j['job_id']!='atomicSystemWide__cuda_jl';assert j['group']=='foreign'
  for p,h in j['readonly_bindings'].items():assert sha(p)==h
  l=json.loads((R/'ontology/studies/v100_roofline_537/benchmarks'/(j['benchmark']+'.json')).read_bytes());assert all(l['accepted_named_contract'][k]==v for k,v in j['expected_contract'].items());assert l['models'][j['model']]['latest_profile_attempt']['profile_id']==j['expected_latest_profile_id'];proof.append({'job_id':j['job_id'],'prepared_plan_sha256':sha(j['prepared_plan']['path']),'full_inverse_to_unattempted065_plan_equal':True,'former065_slot_absent':True,'future072_root_absent':True,'flat_depth_valid':True,'current_contract_and_latest_match':True})
save(O/'source_plan_readonly_validation.json',{'status':'saved_JSON_source_plan_checks_passed_no_runtime_execution','records':proof,'controller_AST_source_parsed_only':True,'import_adapter_AST_source_parsed_only':True,'no_proposed_import_execution_or_GPU_build_tests':True})
print('controller',sha(O/'controller072.py'));print('proposal',sha(O/'proposal072.json'));print('import',sha(O/'import_nested065.py'))
