from pathlib import Path
import json,hashlib,copy,fcntl
O=Path(__file__).resolve().parent;P=O.parent/'tgvnn064_foreign065_recovery_proposal_v2';R=Path('/home/imo/HeCBench_julia');RAW=Path('/tmp/hecbench-roofline-v100-20261020');H=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2)+'\n')
def sub(x,a,b):
 if isinstance(x,str):return x.replace(a,b)
 if isinstance(x,list):return[sub(z,a,b)for z in x]
 if isinstance(x,dict):return{sub(k,a,b):sub(v,a,b)for k,v in x.items()}
 return x
parent=json.loads((P/'controller_foreign_receipt.json').read_bytes());assert parent['collector_invocations']==1 and len(parent['steps'])==1;assert parent['steps'][0]['jobs']==['atomicSystemWide__cuda_jl'];assert parent['status']=='new_foreign_activity_stop_remaining_jobs_unattempted_no_retry';review=json.loads((P/'launch_review.json').read_bytes());jobs=[]
with Path('/tmp/hecbench-roofline-metadata.lock').open('r')as lk:
 fcntl.flock(lk,fcntl.LOCK_SH)
 for j in review['jobs']:
  if j['group']!='foreign'or j['issues']or j['job_id']=='atomicSystemWide__cuda_jl':continue
  oldplan=json.loads(Path(j['prepared_plan']['path']).read_bytes());newroot=RAW/('batch072-'+j['job_id']);assert not newroot.exists();assert not Path(j['future_plan_path']).parent.exists();newplan=sub(oldplan,j['plan_root'],str(newroot));assert sub(newplan,str(newroot),j['plan_root'])==oldplan;new=copy.deepcopy(j)
  for field in ['plan_root','future_plan_path','scratch','argv','source_copies']:new[field]=sub(new[field],j['plan_root'],str(newroot))
  new['former065_prospective_plan']=j['prepared_plan'];new['former065_slot_never_created_at_preparation']=True
  dest=O/'plans'/j['job_id']/'plan.json';save(dest,newplan);new['prepared_plan']={'path':str(dest),'sha256':H(dest),'bytes':dest.stat().st_size};new['full_inverse_to065_prospective_plan_equal']=True
  for p,h in j['readonly_bindings'].items():assert H(p)==h,(p,h)
  ledger=json.loads((R/'ontology/studies/v100_roofline_537/benchmarks'/(j['benchmark']+'.json')).read_bytes());assert all(ledger['accepted_named_contract'][k]==v for k,v in j['expected_contract'].items());assert ledger['models'][j['model']]['latest_profile_attempt']['profile_id']==j['expected_latest_profile_id'];jobs.append(new)
assert len(jobs)==26;assert len({j['job_id']for j in jobs})==26
save(O/'continuation_jobs.json',{'jobs':jobs,'excluded_attempted_atomic':'atomicSystemWide__cuda_jl','held_without_source_recreation':'tsne__ka','parent065_receipt':{'path':str(P/'controller_foreign_receipt.json'),'sha256':H(P/'controller_foreign_receipt.json')},'parent_controller':{'path':str(P/'controller.py'),'sha256':H(P/'controller.py')},'parent_review':{'path':str(P/'launch_review.json'),'sha256':H(P/'launch_review.json')},'no_current072_staging_or_execution':True})
slot=RAW/'batch065/atomicSystemWide__cuda_jl/atomicSystemWide/cuda_jl';files=[]
for p in sorted(slot.iterdir()):
 assert p.is_file()and not p.is_symlink();files.append({'name':p.name,'bytes':p.stat().st_size,'sha256':H(p)})
prov=json.loads((slot/'profile_provenance.json').read_bytes());plan=json.loads((slot/'plan.json').read_bytes());assert prov['status']=='profile_process_failed'and prov['profile']['stop_reason']=='external_gpu_activity_detected_own_profile_stopped'and prov['profile']['returncode']==-15;assert prov.get('finished_at');assert prov['plan_sha256']==H(slot/'plan.json');assert prov['counters_sha256']==H(slot/'counters.csv');assert plan['benchmark']=='atomicSystemWide'and plan['model']=='cuda_jl';assert not (slot/'derived.json').exists()
alias=RAW/'batch065-atomicSystemWide__cuda_jl';assert not alias.exists()
save(O/'nested065_import_inventory.json',{'original_slot':str(slot),'prospective_flat_alias_root':str(alias),'prospective_flat_alias_slot':str(alias/'atomicSystemWide/cuda_jl'),'prospective_frozen_profile':'ontology/studies/v100_roofline_537/profiles/atomicSystemWide/cuda_jl/batch065-atomicSystemWide__cuda_jl','source_status':prov['status'],'source_provenance_sha256':H(slot/'profile_provenance.json'),'source_plan_sha256':H(slot/'plan.json'),'configuration_id':plan['configuration_id'],'raw_config_sha256':plan['raw_config_sha256'],'reference_result_sha256':plan['reference_result_sha256'],'files':files,'file_count':len(files),'byte_total':sum(f['bytes']for f in files),'new_GPU_processes':0,'new_build_processes':0,'purpose':'Lossless flat raw alias import of exactly one finished failed nested attempt; no source/provenance/history rewrites, no new acceptance or position. Original nested source remains unchanged.'})
print('26plans',len(jobs),'importfiles',len(files),'bytes',sum(f['bytes']for f in files))
