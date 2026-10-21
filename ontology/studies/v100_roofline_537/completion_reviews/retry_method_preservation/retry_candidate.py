#!/usr/bin/env python3
"""Private candidate: finite conservative generic retry, never a custom-method copier.
No command execution unless --execute. Holds are preservation dispositions, not
unsupported-framework claims. No active source tool is changed by this file.
"""
import argparse,collections,datetime,fcntl,gzip,hashlib,json,math,os
from pathlib import Path
import subprocess,sys
REASONS={'external_gpu_activity_detected_own_profile_stopped','wall_timeout'}
STANDARD=set('benchmark model configuration_id reference_result reference_result_sha256 raw_config_sha256 effective_config_adjudication outcome qualification source_cwd_readonly_inputs scratch_cwd build_command profile_target_command environment_overrides runtime_assets runtime_asset_bytes execution_bindings source_path_literal_findings review blocking_gates notes staging_status effective_original_timing_region kernel_ROI_mapping_status profiler_requirements commands_executed binding_validation staged'.split())
METHOD_FIELDS=('application_replay','profile_window','isolated_assertion','large_artifact_policy','tensor_mode_bindings_before_profile','tensor_mode_decisions_sha256','selection_override')
def stamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,d):Path(p).write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
def norm(value,plan):
 scratch=plan['scratch_cwd']
 if isinstance(value,str):return value.replace(scratch,'<FRESH_SCRATCH>')
 if isinstance(value,list):return[norm(x,plan)for x in value]
 if isinstance(value,dict):return{norm(k,plan):norm(v,plan)for k,v in value.items()}
 return value
def signature(plan):
 return{k:norm(plan.get(k),plan)for k in ('build_command','profile_target_command','environment_overrides','source_cwd_readonly_inputs','effective_original_timing_region','reference_result_sha256','raw_config_sha256','outcome')}
def frozen(slot,root):
 manifest_path=slot/'manifest.json';manifest=json.loads(manifest_path.read_bytes());files={}
 for name in ['plan.json','profile_provenance.json']:
  matches=[x for x in manifest['files']if x['original_name']==name]
  if len(matches)!=1:raise RuntimeError('Missing/duplicate captured member '+name)
  item=matches[0];path=(root/item['path']).resolve()
  if not path.is_relative_to((root/'ontology/studies/v100_roofline_537').resolve()):raise RuntimeError('Frozen path escape')
  packed=path.read_bytes();raw=gzip.decompress(packed)
  if len(packed)!=item['compressed_bytes']or hashlib.sha256(packed).hexdigest()!=item['compressed_sha256']or len(raw)!=item['uncompressed_bytes']or hashlib.sha256(raw).hexdigest()!=item['uncompressed_sha256']:raise RuntimeError('Frozen member SHA/length mismatch')
  files[name]=(json.loads(raw),hashlib.sha256(raw).hexdigest(),str(path),sha(path))
 plan,ph,_,_=files['plan.json'];prov,_,_,_=files['profile_provenance.json']
 if ph!=prov['plan_sha256']:raise RuntimeError('Captured plan/provenance mismatch')
 return plan,prov,{'manifest':str(manifest_path),'manifest_sha256':sha(manifest_path),'members':{k:{'raw_sha256':v[1],'path':v[2],'compressed_sha256':v[3]}for k,v in files.items()}}
def method_holds(plan,prov):
 holds=[]
 for k in sorted(set(plan)-STANDARD):holds.append('noncanonical_plan_field:'+k)
 for k in METHOD_FIELDS:
  if prov.get(k):holds.append('custom_collector_method:'+k)
 if plan.get('benchmark')in('perlin','zmddft','su3'):holds.append('large_payload_optin_required')
 if plan.get('benchmark')in('assert','atomicSystemWide','streamUM','rayleighBenardConvection'):holds.append('explicit_case_method_review_required')
 return holds

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--source-root',type=Path,required=True);ap.add_argument('--raw-root',type=Path,required=True);ap.add_argument('--batch-id',required=True);ap.add_argument('--reviews',type=Path,required=True);ap.add_argument('--output-root',type=Path,help='Fresh private report directory; defaults to raw-root/retry_ID_method_review');ap.add_argument('--timeout',type=int,default=600);ap.add_argument('--freeze-every',type=int,default=10);ap.add_argument('--execute',action='store_true');a=ap.parse_args()
 root=a.source_root.resolve();raw=a.raw_root.resolve();out=(a.output_root or(raw/('retry_'+a.batch_id+'_method_review'))).resolve();study=root/'ontology/studies/v100_roofline_537'
 if not raw.is_relative_to('/tmp')or raw==Path('/tmp')or not raw.is_dir()or raw.stat().st_uid!=os.getuid():ap.error('Existing owned raw tree under/tmp required')
 if out.exists()or not out.is_relative_to('/tmp')or out==Path('/tmp'):ap.error('Fresh private /tmp output required')
 if a.batch_id not in ('039','041')or not 1<=a.timeout<=3600 or a.freeze_every<1:ap.error('Finite039/041 only; operational requested cap3600; positive freeze interval')
 if any(raw.glob('batch'+a.batch_id+'-*'))or(raw/('retry_'+a.batch_id+'_frontier.json')).exists()or(raw/('retry_'+a.batch_id+'.jsonl')).exists():ap.error('Existing round artifacts: preserve/import, never re-execute')
 out.mkdir();journal=out/'journal.jsonl'
 def note(**r):
  r['time']=stamp()
  with journal.open('a')as f:f.write(json.dumps(r,allow_nan=False)+'\n')
  if a.execute:
   with(raw/('retry_'+a.batch_id+'.jsonl')).open('a')as f:f.write(json.dumps(r,allow_nan=False)+'\n')
  print(json.dumps(r),flush=True)
 def command(label,cmd):
  # Invoked only on explicit root adoption; no shell, no source/OS mutations.
  for p,h in toolbindings.items():
   if sha(p)!=h:raise RuntimeError('Bound generic tool/review changed before command '+p)
  log=out/(label+'.log')
  with log.open('xb')as f:rc=subprocess.run(cmd,cwd=root,stdout=f,stderr=subprocess.STDOUT).returncode
  note(event=label,command=cmd,returncode=rc,log=str(log),log_sha256=sha(log),tool_bindings=toolbindings)
  for p,h in toolbindings.items():
   if sha(p)!=h:raise RuntimeError('Bound generic tool/review changed during command '+p)
  return rc
 records=[];bindings={};frontier={}
 with Path('/tmp/hecbench-roofline-metadata.lock').open('rb')as lock:
  fcntl.flock(lock,fcntl.LOCK_SH)
  for lp in sorted((study/'benchmarks').glob('*.json')):
   ledger=json.loads(lp.read_bytes());bindings[str(lp)]=sha(lp)
   for model,state in ledger['models'].items():
    if state.get('profiling_status')!='profile_process_failed'or not state.get('profile_ids'):continue
    selected=root/state['profile_ids'][0]
    if not(selected/'manifest.json').is_file():continue
    plan,pr,evidence=frozen(selected,root);reason=pr.get('profile',{}).get('stop_reason')
    if reason not in REASONS:continue
    row={'benchmark':ledger['benchmark'],'model':model,'reason':reason,'selected_parent':evidence,'configuration_id':plan['configuration_id'],'historical_method_holds':[],'historical_deadlines':[],'historical_attempts':[],'disposition':'requires_generic_plan_equivalence_before_execute','effective_retry_timeout':None}
    for slot in sorted((study/'profiles'/ledger['benchmark']/model).iterdir()):
     if not(slot/'manifest.json').is_file():continue
     hp,hpr,he=frozen(slot,root);holds=method_holds(hp,hpr)
     if signature(hp)!=signature(plan):holds.append('historical_execution_method_signature_differs_from_selected')
     row['historical_attempts'].append(he)
     if holds:row['historical_method_holds'].append({'profile_id':str(slot.relative_to(root)),'reasons':holds})
     deadline=hpr.get('operational_deadline')or{}
     for field in ['requested_seconds','effective_seconds']:
      x=deadline.get(field)
      if isinstance(x,(int,float))and not isinstance(x,bool)and math.isfinite(x)and x>0:row['historical_deadlines'].append({'profile_id':str(slot.relative_to(root)),'field':field,'value':x})
     if hpr.get('profile')and not deadline:row['historical_method_holds'].append({'profile_id':str(slot.relative_to(root)),'reasons':['historical_deadline_missing_no_invented_bound']})
    bound=math.ceil(max([a.timeout]+[x['value']for x in row['historical_deadlines']]))
    if bound>3600:row['historical_method_holds'].append({'profile_id':None,'reasons':['historical_deadline_exceeds_explicit3600_cap_requires_review']})
    row['effective_retry_timeout']=bound if bound<=3600 else None
    if row['historical_method_holds']:row['disposition']='held_requires_fresh_source_bound_custom_method_reapplication'
    records.append(row);frontier.setdefault(row['benchmark'],[]).append(row)
 report={'schema_version':1,'created_at':stamp(),'source_bindings':bindings,'selected_operational_stop_slots':len(records),'dispositions':dict(collections.Counter(x['disposition']for x in records)),'records':records,'script_sha256':sha(__file__),'active_tools_unchanged':True,'GPU_build_tests_executed_by_review':False,'timeout_policy':'Per slot max(CLI requested, all recorded historical requested/effective seconds), mandatory<=3600; hold if historical full bound cannot be preserved. Collector formula may increase up to3600 using saved primary mean.','scope':'Conservative method-preservation holds, not unsupported frameworks or terminal scientific failures.'}
 dump(out/'method_preservation_review.json',report)
 if not a.execute:return 0
 # Old round namespace is reserved once, even if all held. No iteration repeats.
 reservation=raw/('retry_'+a.batch_id+'_frontier.json')
 with reservation.open('x')as f:json.dump({'source_review':str(out/'method_preservation_review.json'),'sha256':sha(out/'method_preservation_review.json'),'frontier':frontier},f,indent=2);f.write('\n')
 toolbindings={str(root/'tools'/n):sha(root/'tools'/n)for n in ['v100_roofline_prepare.py','v100_roofline_collect.py','v100_roofline_summarize.py']};toolbindings[str(a.reviews.resolve())]=sha(a.reviews);toolbindings[str(Path(__file__).resolve())]=sha(__file__);dump(out/'execution_tool_bindings.json',toolbindings)
 pending=0
 for row in records:
  if row['disposition'].startswith('held_'):continue
  b,m=row['benchmark'],row['model']
  # Recheck selection after lock waits; never retry a newer result/method blindly.
  with Path('/tmp/hecbench-roofline-metadata.lock').open('rb')as lock:
   fcntl.flock(lock,fcntl.LOCK_SH);state=json.loads((study/'benchmarks'/(b+'.json')).read_text())['models'][m]
   parent_manifest=Path(row['selected_parent']['manifest'])
   if state.get('profiling_status')!='profile_process_failed'or not state.get('profile_ids')or root/state['profile_ids'][0]!=parent_manifest.parent:
    row['disposition']='skipped_selection_changed';continue
  for p,h in toolbindings.items():
   if sha(p)!=h:raise RuntimeError('Execution tool/review bytes drifted')
  plans=raw/('batch'+a.batch_id+'-retry-'+b+'-'+m)
  cmd=[sys.executable,str(root/'tools/v100_roofline_prepare.py'),'--source-root',str(root),'--benchmark',b,'--models',m,'--output-root',str(plans),'--reviews',str(a.reviews.resolve()),'--asset-cap-mib','1024','--stage']
  if command('prepare_'+b+'_'+m,cmd):row['disposition']='prepare_failed_preserved';continue
  freshpath=plans/b/m/'plan.json'
  if not freshpath.is_file():row['disposition']='missing_fresh_plan_held';continue
  fresh=json.loads(freshpath.read_text());old,_,_=frozen(parent_manifest.parent,root);diff=[]
  if signature(fresh)!=signature(old):diff.append('fresh_generic_execution_semantics_differ_from_selected')
  if fresh.get('blocking_gates')or not fresh.get('staged'):diff.append('fresh_plan_not_staged_or_gated')
  # Parent's bound file roles/dependencies cannot silently disappear/change.
  if norm(old.get('execution_bindings'),old)!=norm(fresh.get('execution_bindings'),fresh):diff.append('fresh_execution_binding_set_or_hash_differs_requires_review')
  if diff:row['disposition']='held_fresh_generic_plan_not_proven_equivalent';row['fresh_equivalence_holds']=diff;continue
  # No source-selected method or review flags are dropped: such cases held above.
  collect=[sys.executable,str(root/'tools/v100_roofline_collect.py'),'--source-root',str(root),'--plan-root',str(plans),'--timeout',str(row['effective_retry_timeout'])]
  if row['reason']=='wall_timeout':
   selector=plans/'bounded_selector.json';dump(selector,{'benchmark':b,'configuration_id':fresh['configuration_id'],'fixture_description':{'path':str(reservation),'sha256':sha(reservation)},'expected_matching_launches':None,'selection':{m:['--filter-mode','global','--kernel-name-base','mangled','--kernel-id','::regex:.*:1']},'coverage':'First invocation per mangled specialization across geometries; explicit geometry/state/phase undercoverage, not complete ROI.','parent_failures':[row['selected_parent']]});collect+=['--selection-overrides',str(selector)]
  row['collect_launcher_returncode']=command('collect_'+b+'_'+m,collect);row['disposition']='retry_launcher_finished_inspect_outcome';pending+=1
  dump(out/'execution_dispositions.json',records)
  if pending>=a.freeze_every:
   if command('freeze_'+b+'_'+m,[sys.executable,str(root/'tools/v100_roofline_summarize.py'),'--source-root',str(root),'--raw-root',str(raw)]):raise RuntimeError('Freeze failed')
   pending=0
 if pending:
  if command('freeze_final',[sys.executable,str(root/'tools/v100_roofline_summarize.py'),'--source-root',str(root),'--raw-root',str(raw)]):raise RuntimeError('Final freeze failed')
 dump(out/'execution_dispositions.json',records);note(event='retry_traversal_finished',meaning='Finite preservation-aware traversal only; held methods remain pending explicit reapplication, no scientific/ROI completion claim')
 return 0
if __name__=='__main__':raise SystemExit(main())
