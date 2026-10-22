# Independent stdlib-only saved-document comparisons; never imports/executes proposed tools.
from pathlib import Path
import json,gzip,hashlib,fcntl,collections,copy,datetime
O=Path(__file__).resolve().parent;P=Path('/tmp/roofline-completion-20261021/tgvnn064_foreign065_recovery_proposal_v2');R=Path('/home/imo/HeCBench_julia');H=lambda b:hashlib.sha256(b).hexdigest()
def sha(p):return H(Path(p).read_bytes())
def load(p):
 p=Path(p);return json.loads(gzip.decompress(p.read_bytes())if p.suffix=='.gz'else p.read_bytes())
def back(v,new,old):
 if isinstance(v,str):return v.replace(new,old)
 if isinstance(v,list):return [back(z,new,old)for z in v]
 if isinstance(v,dict):return {back(k,new,old):back(z,new,old)for k,z in v.items()}
 return v
rev=load(P/'launch_review.json');assert rev['controller_sha256']=='4837cb70cd41b92947b672322c3729e285781706db4c0f8454b80c5bcb1eac07';assert sha(P/'controller.py')==rev['controller_sha256'];assert sha(P/'launch_review.json')=='087e9285b57945b8f302035edc164775050508f83c62c906d127148c6193f99c'
checks=[];bindings={**rev['tool_bindings'],**rev['packet_bindings']};records=[]
with Path('/tmp/hecbench-roofline-metadata.lock').open('r')as lk:
 fcntl.flock(lk,fcntl.LOCK_SH)
 ledgers={j['benchmark']:load(R/'ontology/studies/v100_roofline_537/benchmarks'/(j['benchmark']+'.json'))for j in rev['jobs']}
for job in rev['jobs']:
 old=load(job['original_plan']['path']);new=load(job['prepared_plan']['path']);prov=load(job['previous_profile']['path']);assert sha(job['original_plan']['path'])==job['original_plan']['sha256'];assert sha(job['prepared_plan']['path'])==job['prepared_plan']['sha256'];assert sha(job['previous_profile']['path'])==job['previous_profile']['sha256'];inverse=copy.deepcopy(new)
 cps=[]
 for c in job['source_copies']:
  copied=Path(c['frozen_copy']['path']).read_bytes();assert H(copied)==c['original_sha256']==c['future_sha256'];assert copied==Path(c['original']).read_bytes();assert job['old_scratch'].encode()not in copied;assert Path(c['future_copy']).is_relative_to(job['scratch']);assert inverse['execution_bindings'].pop(c['future_copy'])==c['future_sha256'];cps.append(c['original'])
 inverse=back(inverse,job['scratch'],job['old_scratch']);assert inverse==old
 ct=ledgers[job['benchmark']]['accepted_named_contract'];assert all(ct[k]==v for k,v in job['expected_contract'].items());assert new['configuration_id']==old['configuration_id']==ct['configuration_id'];assert new['raw_config_sha256']==old['raw_config_sha256']==ct['raw_config_sha256'];assert new['reference_result_sha256']==old['reference_result_sha256']==ct['result_sha256'];assert new['blocking_gates']==old['blocking_gates'];assert new['environment_overrides']['CUDA_VISIBLE_DEVICES']=='6';assert new['scratch_cwd']==job['scratch'];assert str(Path(job['scratch']).resolve()).startswith('/tmp/hecbench-roofline-v100-20261020/batch06')
 actualcap=Path(job['previous_profile']['path']).parent;assert load(actualcap/'plan.json.gz')==old
 if not job['issues']:
  for p,h in job['readonly_bindings'].items():assert sha(p)==h,(job['job_id'],p);bindings[p]=h
  assert ledgers[job['benchmark']]['models'][job['model']]['latest_profile_attempt']['profile_id']==job['expected_latest_profile_id']
 if job['group']=='foreign':assert prov['profile']['stop_reason']=='external_gpu_activity_detected_own_profile_stopped';assert job['timeout_seconds']==prov['operational_deadline']['requested_seconds']
 else:assert prov['profile']['stop_reason']=='wall_timeout';assert job['timeout_seconds']==3600;assert prov['operational_deadline']['requested_seconds']==600
 assert '--retain-auxiliary'not in job['argv'];assert '--skip-qualified-selected'in job['argv'];assert '--source-root'in job['argv'];assert job['argv'][job['argv'].index('--source-root')+1]==str(R)
 if job['method']:
  assert job['method']==prov['profiler_method'];assert '--profiler-executable'in job['argv'];assert '--profiler-method-review'in job['argv'];assert '--profiler-method-review-sha256'in job['argv']
 if prov.get('application_replay'):assert '--application-replay-reviewed'in job['argv']
 if job['selection']:
  sel=load(job['selection']['path']);assert sha(job['selection']['path'])==job['selection']['sha256'];oldsel=load(prov['selection_override']['path']);assert sel['selection'][job['model']]==oldsel['selection'][job['model']]
  for k in ['benchmark','configuration_id','fixture_description','expected_matching_launches','coverage']:assert sel[k]==oldsel[k],(job['job_id'],k)
  assert job['argv'][job['argv'].index('--selection-overrides')+1]==job['selection']['path']
 records.append({'job_id':job['job_id'],'eligible':not bool(job['issues']),'inverse_full_json_equal':True,'same_frozen_contract':True,'same_saved_model_selector':True,'byte_identical_host_copies':cps,'timeout_seconds':job['timeout_seconds'],'prepared_plan_sha256':job['prepared_plan']['sha256']})
for p,h in bindings.items():assert sha(p)==h
v1=load(P.parent/'tgvnn064_foreign065_recovery_proposal/launch_review.json');assert rev['predecessors']==v1['predecessors'];assert len(rev['predecessors'])==43
assert len({j['job_id']for j in rev['jobs']})==32;assert sum(r['eligible']and r['job_id'].startswith('tgvnn__')for r in records)==4;assert sum(r['eligible']and not r['job_id'].startswith('tgvnn__')for r in records)==27
summary={'status':'independent_saved_data_checks_passed_source_review_pending_root_adoption','controller_sha256':sha(P/'controller.py'),'launch_review_sha256':sha(P/'launch_review.json'),'jobs':32,'eligible_foreign':27,'eligible_tgvnn':4,'held':['tsne__ka'],'unchanged_predecessor_generations':43,'unique_eligible_current_source_tool_evidence_bindings':len(bindings),'exact_selector_mappings_and_fresh_inverse_plans_passed':True,'no_controller_or_target_import_execution':True,'no_GPU_queries_or_runs':True,'no_builds_or_tests':True,'created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'records':records}
(O/'independent_saved_data_audit.json').write_text(json.dumps(summary,indent=2)+'\n');print({k:v for k,v in summary.items()if k!='records'})
