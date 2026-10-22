from pathlib import Path
import json,hashlib,gzip,fcntl,datetime,shutil
O=Path(__file__).parent.resolve();W=Path('/home/imo/HeCBench_julia');H=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();d=json.loads((O/'plans.json').read_bytes());checks=[];issues=[]
def ck(name,value,detail):
 checks.append({'check':name,'passed':bool(value),'detail':detail})
 if not value:issues.append(checks[-1])
with open('/tmp/hecbench-roofline-metadata.lock','r')as l:
 fcntl.flock(l,fcntl.LOCK_SH)
 for row in d['plans']:
  p=json.loads(Path(row['plan']['path']).read_bytes());old=json.loads(Path(row['plan']['path']).with_name('original_plan.json').read_bytes());b,m=row['benchmark'],row['model'];tag=b+'/'+m
  ck('plan_binding',H(row['plan']['path'])==row['plan']['sha256'],tag)
  ledger=json.loads((W/'ontology/studies/v100_roofline_537/benchmarks'/f'{b}.json').read_bytes());contract=ledger['accepted_named_contract'];ck('current_contract',all(p[a]==contract[z]for a,z in [('configuration_id','configuration_id'),('reference_result_sha256','result_sha256'),('raw_config_sha256','raw_config_sha256')]),tag)
  hist=gzip.decompress(Path(row['historical_plan']['path']).read_bytes());ck('retained_historical_plan',json.loads(hist)==old,tag)
  for path,h in p['execution_bindings'].items():ck('binding_unchanged',H(path)==h,{'slot':tag,'path':path})
  ck('numeric_plan_UUID_runtime_contract',p['environment_overrides']['CUDA_VISIBLE_DEVICES']=='6',tag)
  for patch in p['capture_overlay_provenance']['patches']:
   src=Path(patch['source']).read_bytes();out=Path(patch['copy']).read_bytes();ck('copy_source_actual_SHA',hashlib.sha256(src).hexdigest()==patch['source_sha256']and hashlib.sha256(out).hexdigest()==patch['copy_sha256'],tag)
   inverse=out.decode()
   if patch['whole_guard_replacement']:
    ck('whole_guard_only_component',Path(patch['source']).name=='study_guard.jl'and src.decode()==patch['original_text'],tag);inverse=patch['original_text']
   else:
    for pair in reversed(patch['literal_substitutions']):
     ck('unique_inverse_substitution',inverse.count(pair['after'])==1,tag);inverse=inverse.replace(pair['after'],pair['before'],1)
   ck('exact_inverse_original_bytes',inverse.encode()==src,tag)
  ck('timing_contract_unchanged',p['effective_original_timing_region']==old['effective_original_timing_region'],tag)
  ck('no_gate_removal',p['blocking_gates']==old['blocking_gates']==[],tag)
  if b=='testSNAP':
   common=next(x for x in p['capture_overlay_provenance']['patches']if Path(x['source']).name=='study_common.jl');ck('only_ROOT_capture_substitutions',len(common['literal_substitutions'])==2,tag);ck('observer_CPP_unchanged',p['build_command'][4]==old['build_command'][4],tag)
  else:
   ck('only_single_device_primary',p['profile_target_command'][-1]=='1',tag)
   for x in p['capture_overlay_provenance']['patches']:
    ck('copied_no_GPU5_query','nvidia-smi -i 5'not in x['text'] and '"5,6"'not in x['text'],{'slot':tag,'copy':x['copy']})
   ck('UUID_explicit_in_guard',any('GPU-c122ecbd-d9f3-18a5-42d4-1535af37e56d'in x['text']for x in p['capture_overlay_provenance']['patches']),tag)
 fcntl.flock(l,fcntl.LOCK_UN)
free=shutil.disk_usage('/tmp').free;payload={'testSNAP_Julia3':17715669*8*5*3,'simpleMultiDevice_all4':33761316*8*5*4};budget={'checked_payload_bytes':payload,'total_checked_payload_bytes':sum(payload.values()),'warmup_payloads_extra':0,'warmup_observer_auxiliary':'testSNAP trial0 still emits full guards/CAS/inputs and must be retained; not included in primary payload arithmetic','auxiliary_allowance_proposal_bytes':7*256*1024**2,'reserve_bytes':2048*1024**2,'current_tmp_free_bytes':free,'estimated_safe_campaign_free_bytes':sum(payload.values())+7*256*1024**2+2048*1024**2,'launch_capacity_hold':free<sum(payload.values())+7*256*1024**2+2048*1024**2,'reason':'Full outputs mandatory. Conservative auxiliary allowance is prospective planning, not a new collector policy. Root must arrange adequate authorized fresh storage or separately review strict finite storage strategy; no deletion/truncation/guard widening proposed.'};(O/'storage_budget.json').write_text(json.dumps(budget,indent=2)+'\n')
review={'status':'PRIVATE_SOURCE_REVIEW_READY_ROOT_ADOPTION_REQUIRED','checks_total':len(checks),'checks_passed':sum(x['passed']for x in checks),'blockers':issues,'checks':checks,'plans':7,'kernel_math_change':False,'timer_or_numeric_contract_change':False,'all_five_original_checks_mandatory':True,'GPU5_use_or_query_authorized':False,'runtime_fix_observed':False,'tests_or_builds_or_GPU_or_target_imports':0,'storage_hold':budget['launch_capacity_hold'],'qualifications':['Guard checks are host preflight/after checks outside original timed windows, not kernel changes.','Julia env-only guard before CUDA load; exact runtime UUID/count assertion after CUDA device enumeration before allocation/warmup.','Native count/UUID guard runs before setup and original per-trial guard callsites outside timer.','Canonical plan keeps CUDA_VISIBLE_DEVICES=6; exact adopted collector replaces it with fixed GPU6 UUID for all child processes.','Existing original CLI/multi-device capability not broadened: selected named g1 argv1 only. GPU5 branch removed only in owned copied guard.','No actual profiler sampled phase/full ROI completeness/causal claim.','No success claimed until new full checked profiling completes; historical failed plans/logs remain unchanged.','One metadata-only staging TypeError and nonexistent proposed compiler path preserved/described; neither target compiler nor GPU attempt occurred.']};(O/'review.json').write_text(json.dumps(review,indent=2)+'\n');print(json.dumps({'checks':len(checks),'issues':issues,'storage_hold':budget['launch_capacity_hold'],'review_sha256':H(O/'review.json')}))
