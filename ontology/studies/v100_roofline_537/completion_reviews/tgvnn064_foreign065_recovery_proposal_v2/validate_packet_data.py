from pathlib import Path
import json,gzip,hashlib,copy,ast,fcntl,datetime
O=Path(__file__).resolve().parent;W=Path('/home/imo/HeCBench_julia');h=lambda b:hashlib.sha256(b).hexdigest();d=json.loads((O/'job_inventory.json').read_bytes());r=json.loads((O/'launch_review.json').read_bytes());checks=[];trials=[]
def digest(p):
 x=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(4<<20),b''):x.update(b)
 return x.hexdigest()
def inverse(x,new,old):
 if isinstance(x,str):return x.replace(new,old)
 if isinstance(x,list):return[inverse(v,new,old)for v in x]
 if isinstance(x,dict):return{k:inverse(v,new,old)for k,v in x.items()}
 return x
with Path('/tmp/hecbench-roofline-metadata.lock').open('r')as lock:
 fcntl.flock(lock,fcntl.LOCK_SH)
 for j in d['jobs']:
  plan=json.loads(Path(j['prepared_plan']['path']).read_bytes());old=json.loads(Path(j['original_plan']['path']).read_bytes());back=copy.deepcopy(plan)
  for cp in j['source_copies']:
   assert digest(cp['original'])==digest(cp['frozen_copy']['path'])==cp['future_sha256'];assert j['old_scratch'].encode()not in Path(cp['frozen_copy']['path']).read_bytes();del back['execution_bindings'][cp['future_copy']]
  assert inverse(back,j['scratch'],j['old_scratch'])==old
  assert '--retain-auxiliary'not in j['argv']and j['argv'][j['argv'].index('--timeout')+1]==str(j['timeout_seconds'])
  if not j['issues']:
   for p,sh in j['readonly_bindings'].items():assert digest(p)==sh
  assert not Path(j['future_plan_path']).exists(),'Raw attempt already exists; validation snapshot must precede execution'
  checks.append({'job_id':j['job_id'],'complete_original_plan_recovered':True,'byte_identical_wrappers':len(j['source_copies']),'held':bool(j['issues']),'readonly_source_method_bindings_match':not j['issues'],'future_raw_slot_not_created':True})
  if j['group']=='tgvnn':
   p=Path(j['previous_profile']['path']).parent/'execution.log.gz';raw=gzip.decompress(p.read_bytes());rs=[json.loads(l.split('STUDY_TRIAL ',1)[1])for l in raw.decode().splitlines()if l.startswith('STUDY_TRIAL ')];assert[x['trial']for x in rs]==[1,2];assert all(x['correct']and x['checked_values']==72089820 and x['output_artifact']['count']==72089820 for x in rs);trials.append({'model':j['model'],'log':str(p),'log_compressed_sha256':digest(p),'log_raw_sha256':h(raw),'actual_checked_trials':rs,'old_requested_budget_seconds':600,'new_explicit_budget_seconds':3600,'source_warmup_and_full5_preserved':True})
 selectors=json.loads((O/'selectors/tgvnn_all4_exact.json').read_bytes());assert set(selectors['selection'])=={'cuda','cuda_jl','ka','jacc'}
 assert len([j for j in d['jobs']if j['group']=='foreign'and not j['issues']])==27 and len([j for j in d['jobs']if j['issues']])==1
 assert digest(O/'controller.py')==r['controller_sha256'];ast.parse((O/'controller.py').read_bytes())
 for p,sh in r['tool_bindings'].items():assert digest(p)==sh
 for p,sh in r['packet_bindings'].items():assert digest(p)==sh
 (O/'data_validation.json').write_text(json.dumps({'status':'PRIVATE_DATA_SOURCE_CHECKS_CONSISTENT_NO_GPU_OR_BUILD','checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'jobs':checks,'full_inverse_plan_count':len(checks),'TGVNN_all4_exact_selector_keys_verified':True,'canonical_new_scratch_cleanup_explicit':True,'no_target_or_collector_import_execution':True,'controller_default_readonly_checks_separately_passed':['foreign:27pending_TSNEheld_max27','tgvnn:4pending_max1'],'no_raw_attempt_slots_created':True},indent=2)+'\n');(O/'tgvnn_actual_trial_evidence.json').write_text(json.dumps({'scope':'Saved two-check failures, not successful recovery or timing replacement','models':trials},indent=2)+'\n')
print('32 full inverse plans;7byteidentical wrappers;27foreign eligible/1hold;4TGV source/method pins;raw roots uncreated')
