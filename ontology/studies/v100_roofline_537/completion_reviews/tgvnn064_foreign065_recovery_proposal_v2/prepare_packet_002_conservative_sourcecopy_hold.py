import json,gzip,hashlib,fcntl,copy,datetime,os,shutil
from pathlib import Path
W=Path('/home/imo/HeCBench_julia');S=W/'ontology/studies/v100_roofline_537';O=Path(__file__).resolve().parent;RAW=Path('/tmp/hecbench-roofline-v100-20261020');T=O.parent
cache={}
def digest(p):
 p=Path(p);st=p.stat();key=(str(p),st.st_size,st.st_mtime_ns)
 if key not in cache:
  h=hashlib.sha256()
  with p.open('rb')as f:
   for b in iter(lambda:f.read(4<<20),b''):h.update(b)
  cache[key]=h.hexdigest()
 return cache[key]
def load(p):p=Path(p);return json.loads(gzip.decompress(p.read_bytes())if p.suffix=='.gz'else p.read_bytes())
def info(p):p=Path(p);return {'path':str(p),'sha256':digest(p),'bytes':p.stat().st_size}
def save(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2)+'\n')
def move(x,old,new):
 if isinstance(x,str):return x.replace(old,new)
 if isinstance(x,list):return [move(z,old,new)for z in x]
 if isinstance(x,dict):return {k:move(v,old,new)for k,v in x.items()}
 return x
rows=load(T/'final_v19_remaining_gap_triage/triage_all114.json')['slots'];rows=[r for r in rows if r['category']=='observed_foreign_gpu_stop'or r['benchmark']=='tgvnn'];assert len(rows)==32
jobs=[];allbind={};inspections=[];held=[]
with Path('/tmp/hecbench-roofline-metadata.lock').open('r')as lock:
 fcntl.flock(lock,fcntl.LOCK_SH)
 for r in rows:
  n,m=r['benchmark'],r['model'];group='tgvnn'if n=='tgvnn'else'foreign';lp=S/'benchmarks'/f'{n}.json';l=load(lp);state=l['models'][m];pid=r['latest_profile_id'];cap=W/pid;plan=load(cap/'plan.json.gz');prov=load(cap/'profile_provenance.json.gz');ct=l['accepted_named_contract'];problems=[]
  assert plan['configuration_id']==ct['configuration_id']and plan['raw_config_sha256']==ct['raw_config_sha256']and plan['reference_result_sha256']==ct['result_sha256'];assert state['latest_profile_attempt']['profile_id']==pid
  if group=='foreign':assert prov['profile']['stop_reason']=='external_gpu_activity_detected_own_profile_stopped'
  else:assert prov['profile']['stop_reason']=='wall_timeout'and prov['operational_deadline']['requested_seconds']==600
  binding={str(Path(k).resolve()):v for k,v in plan['execution_bindings'].items()};binding.update({str(Path(k).resolve()):v for k,v in prov.get('additional_current_profile_input_bindings',{}).items()})
  for p,h in binding.items():
   if not Path(p).is_file():problems.append({'path':p,'expected':h,'state':'missing'})
   elif digest(p)!=h:problems.append({'path':p,'expected':h,'actual':digest(p),'state':'changed'})
  old=plan['scratch_cwd'];jobid=n+'__'+m;newroot=RAW/('batch064'if group=='tgvnn'else'batch065');planroot=newroot if group=='tgvnn'else newroot/jobid;slot=planroot/n/m;scratch=slot/'scratch';fresh=move(plan,old,str(scratch))
  # No source binding is renamed: any private executable/source below old scratch
  # needs exact separate source-overlay approval rather than silent recreation.
  for p,h in binding.items():
   if str(Path(p)).startswith(old+'/'):problems.append({'path':p,'state':'old_scratch_source_needs_explicit_relocation_review'})
  for asset in plan.get('runtime_assets',[]):
   for sf in asset['source_files']:
    src=Path(sf['path'])
    if not src.is_file()or digest(src)!=sf['sha256']:problems.append({'path':str(src),'state':'runtime_source_missing_or_changed'})
  assert move(fresh,str(scratch),old)==plan
  timeout=3600 if group=='tgvnn'else prov['operational_deadline']['requested_seconds'];selector=None;extra=[];method=prov.get('profiler_method');profiler=Path(prov['profiler_executable']);toolbindings={str(profiler):prov['profiler_executable_sha256']}
  if method:
   rev=S/'profiler_methods'/method['method_id']/'root_review.json'
   if not rev.exists():
    candidates=[p for p in (S/'profiler_methods'/method['method_id']).glob('*.json')if digest(p)==method['review_sha256']];assert len(candidates)==1;rev=candidates[0]
   assert digest(rev)==method['review_sha256'];rd=load(rev);bundle=rev.parent/rd['bundle_file'];assert digest(bundle)==method['bundle_sha256'];toolbindings.update({str(rev):digest(rev),str(bundle):digest(bundle)});toolbindings.update({rd['profiler']['target_ELF']:rd['profiler']['target_ELF_sha256']});extra+=['--profiler-executable',str(profiler),'--profiler-method-review',str(rev),'--profiler-method-review-sha256',method['review_sha256']]
  else:assert str(profiler)=='/usr/lib/nsight-compute/ncu'and prov['profiler_version'].splitlines()[2].startswith('Version 2022.4.1.0')
  for p,h in toolbindings.items():assert Path(p).is_file()and digest(p)==h
  if prov.get('selection_override'):
   e=prov['selection_override'];sp=Path(e['path']);assert digest(sp)==e['sha256'];selector=O/'selectors'/f'{jobid}.json';selector.parent.mkdir(exist_ok=True);selector.write_bytes(sp.read_bytes());extra+=['--selection-overrides',str(selector)];toolbindings[str(selector)]=digest(selector)
  if prov.get('application_replay'):extra+=['--application-replay-reviewed']
  if prov.get('profile_window')or prov.get('isolated_assertion_phases'):problems.append({'state':'special_phase_requires_separate_explicit_job_review'})
  for label in ['application_replay_review','profile_window_review','isolated_assertion_review']:
   if label in plan:
    e=plan[label];assert digest(e['path'])==e['sha256'];toolbindings[e['path']]=e['sha256']
  # Exact historic plan/provenance are read-only evidence, not overwritten.
  frozen=O/'plans'/jobid/'plan.json';save(frozen,fresh);before=O/'original_plans'/f'{jobid}.json';save(before,plan)
  sources={**binding,**toolbindings};sources[str(cap/'plan.json.gz')]=digest(cap/'plan.json.gz');sources[str(cap/'profile_provenance.json.gz')]=digest(cap/'profile_provenance.json.gz')
  row={'job_id':jobid,'group':group,'benchmark':n,'model':m,'ledger_at_preparation':info(lp),'expected_contract':{k:ct[k]for k in ['configuration_id','raw_config_sha256','result_sha256']},'expected_latest_profile_id':pid,'previous_profile':info(cap/'profile_provenance.json.gz'),'original_plan':info(before),'prepared_plan':info(frozen),'plan_root':str(planroot),'future_plan_path':str(slot/'plan.json'),'scratch':str(scratch),'old_scratch':old,'timeout_seconds':timeout,'method':method,'profiler':{'path':str(profiler),'sha256':digest(profiler),'version':prov['profiler_version']},'selection':info(selector)if selector else None,'collector_extra_arguments':extra,'readonly_bindings':sources,'issues':problems,'full_inverse_plan_equal':True,'plan_mutation_scope':'Literal old scratch prefix to new owned scratch prefix only; scientific/result/config/source/selector/protocol unchanged.'}
  row['argv']=['/usr/bin/python3',str(W/'tools/v100_roofline_collect.py'),'--source-root',str(W),'--plan-root',str(planroot),'--benchmark',n,'--skip-qualified-selected','--timeout',str(timeout),'--reserve-mib','2048','--retain-auxiliary',*extra]
  jobs.append(row);inspections.append({k:row[k]for k in ['job_id','group','issues','method','timeout_seconds','full_inverse_plan_equal']})
  if problems:held.append(jobid)
 for p in ['tools/v100_roofline_collect.py','tools/v100_roofline_classify.py','tools/v100_profile_write_guard.py','tools/v100_profiler_method.py']:
  allbind[str(W/p)]=digest(W/p)
 assert allbind[str(W/'tools/v100_roofline_collect.py')]=='c21b1edd4c6e0924250b306e1b97ff7ff1371b8407eb92899e089dbdca364116'
 save(O/'job_inventory.json',{'jobs':jobs,'tool_bindings':allbind,'triage_input':info(T/'final_v19_remaining_gap_triage/triage_all114.json'),'held_jobs':held,'no_GPU_queries_or_runs':True,'source_method_identity_only_not_runtime_guarantee':True})
 save(O/'inspection_summary.json',{'rows':inspections,'foreign_total':28,'tgvnn_total':4,'held_jobs':held,'eligible_foreign':[r['job_id']for r in jobs if r['group']=='foreign'and not r['issues']],'eligible_tgvnn':[r['job_id']for r in jobs if r['group']=='tgvnn'and not r['issues']]})
print('JOBS',len(jobs),'HELD',held);print('ELIGIBLE_FOREIGN',[r['job_id']for r in jobs if r['group']=='foreign'and not r['issues']])
