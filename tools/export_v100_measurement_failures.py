#!/usr/bin/env python3
"""Export frozen V100 profiling operational outcomes; never translation verdicts.

Live metadata read under coordinator LOCK_SH. All gzip manifest sizes/SHAs,
provenance, archived selected contracts and census joins are checked. Writes
only a fresh /tmp output. No GPU, subprocess, compiler, parser execution or tests.
"""
import argparse,collections,csv,datetime,fcntl,gzip,hashlib,io,json,re,sys
from pathlib import Path
MODELS=('cuda','cuda_jl','ka','jacc')
UUID='GPU-c122ecbd-d9f3-18a5-42d4-1535af37e56d'
PATTERNS={
 'filesystem_permission_diagnostic':r'Permission(?:Error| denied)|Operation not permitted',
 'output_io_diagnostic':r'rawcapturewrite|STUDY_ERROR outputwrite|\btmpfile\b|failed to (?:open|write).*output',
 'dependency_or_loader_diagnostic':r'ModuleNotFoundError|ImportError:|could not load library|error while loading shared libraries|undefined symbol|OpenSSL[^\n]{0,80}legacy provider',
 'missing_path_diagnostic':r'FileNotFoundError|No such file or directory',
 'compiler_child_nonzero_diagnostic':r'(?:g\+\+|nvcc|gcc|clang)[^\n]{0,900}returned non-zero exit status|fatal error:|compilation terminated',
 'helper_subprocess_nonzero_diagnostic':r'CalledProcessError|subprocess\.CalledProcessError|failed process: Process\(',
 'assertion_wrapper_rejection_diagnostic':r'actual isolated assertion status mismatch',
 'instrumented_checker_rejection_diagnostic':r'STUDY_TRIAL [^\n]*"correct"\s*:\s*false|independent[^\n]{0,100}(?:failed|rejected)|oracle[^\n]{0,100}(?:failed|rejected)',
 'payload_truncation_diagnostic':r'payload[^\n]{0,100}truncat|truncated[^\n]{0,100}payload|File size limit exceeded|file size limit exceeded',
 'csv_field_limit_diagnostic':r'field larger than field limit',
 'no_profiled_kernels_diagnostic':r'No kernels were profiled',
 'source_binding_gate_diagnostic':r'(?:Source|source|Input|input|Binding|binding)[^\n]{0,80}(?:changed|mismatch|drift)',
}
RX={k:re.compile(v,re.I if k=='payload_truncation_diagnostic'else 0)for k,v in PATTERNS.items()}
def digest(data):return hashlib.sha256(data).hexdigest()
def sha_file(path):
 h=hashlib.sha256()
 with Path(path).open('rb')as f:
  for b in iter(lambda:f.read(4<<20),b''):h.update(b)
 return h.hexdigest()
def stamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def dump(path,obj):Path(path).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
def countdict(values):return dict(sorted(collections.Counter(values).items()))
def classify(status,prov,observations,na=False):
 if na:return 'source_reviewed_runtime_no_kernel_not_applicable'
 if status=='native_hardware_blocked':return 'native_hardware_barrier'
 if status.startswith('not_started_replay'):return 'replay_method_hold'
 if status.startswith('not_started_plan'):return 'plan_isolation_or_device_gate'
 if status in ['pending_profile_collection','not_collected','waiting_for_idle_gpu','started']:return 'pending_or_active_not_completed'
 if status.startswith('roofline_not_applicable'):return 'source_reviewed_runtime_no_kernel_not_applicable'
 if status=='profile_completed_quality_unverified':return 'completed_process_quality_unverified'
 if status=='profile_process_completed':return 'completed_counter_observation'
 p=prov.get('profile',{});b=prov.get('build',{});stop=p.get('stop_reason')or b.get('stop_reason');error=str(prov.get('error',''))+' '+str(prov.get('counter_parse_error',''))
 if stop and 'external_gpu' in stop or 'existing compute process' in error:return 'foreign_gpu_contention'
 if stop=='wall_timeout':return 'operational_wall_timeout'
 if stop and ('reserve'in stop or 'space'in stop)or 'Free space'in error:return 'storage_reserve_gate'
 if stop and 'payload'in stop:return 'payload_budget_stop'
 if prov.get('large_artifact_validation',{}).get('all_declared_payload_files_complete')is False or 'artifact_incomplete'in status:return 'observed_payload_incomplete'
 if b.get('returncode')is not None and b['returncode']!=0 or 'build failed'in error.lower():return 'build_launcher_failure'
 if 'field larger than field limit'in error:return 'csv_field_size_limit'
 if 'payload_truncation_diagnostic'in observations:return 'payload_truncation_diagnostic_cause_unresolved'
 if 'source_binding_gate_diagnostic'in observations:return 'source_binding_gate'
 for k in ['filesystem_permission_diagnostic','dependency_or_loader_diagnostic','missing_path_diagnostic','compiler_child_nonzero_diagnostic','assertion_wrapper_rejection_diagnostic','output_io_diagnostic','helper_subprocess_nonzero_diagnostic','instrumented_checker_rejection_diagnostic']:
  if k in observations:return k
 if status=='profile_completed_counter_derivation_failed':return 'successful_process_counter_derivation_missing_or_failed'
 if p.get('returncode')not in [None,0]:return 'profile_process_nonzero_cause_unknown'
 return 'operational_cause_unknown'

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--source-root',type=Path,default=Path(__file__).resolve().parents[1]);ap.add_argument('--output-root',type=Path,required=True);ap.add_argument('--lock-path',type=Path,default=Path('/tmp/hecbench-roofline-metadata.lock'));ap.add_argument('--usage-receipt',type=Path);a=ap.parse_args();root=a.source_root.resolve();study=root/'ontology/studies/v100_roofline_537';out=a.output_root.resolve()
 if out.exists()or not out.is_relative_to('/tmp')or out==Path('/tmp'):ap.error('Require nonexistent fresh output strictly under /tmp')
 if not a.lock_path.is_file():ap.error('Coordinator metadata lock must already exist; no implicit unlocked mode')
 inputs={};issues=[];qualifications=[];attempts=[];ledgers={};contracts={};state_snapshot=[];now=stamp();integrity_bytes=0
 def rel(p):return str(Path(p).resolve().relative_to(root))
 def read(path):
  p=Path(path).resolve()
  if not p.is_relative_to(root):raise ValueError('Input escapes source root '+str(p))
  data=p.read_bytes();inputs[rel(p)]=digest(data);return data
 def require(ok,message):
  if not ok:raise ValueError(message)
 def gz(path):
  data=read(path);return gzip.decompress(data)
 snapshots=None
 def source_snapshot(collector_sha):
  nonlocal snapshots
  if snapshots is None:
   snapshots={}
   for p in sorted((study/'tool_snapshots').glob('*.py'))+sorted((study/'tool_snapshots').glob('*.py.gz')):
    data=gz(p)if p.suffix=='.gz'else read(p);snapshots[digest(data)]=data
  return snapshots.get(collector_sha)
 with a.lock_path.open('rb')as lock:
  fcntl.flock(lock,fcntl.LOCK_SH);locked_at=stamp()
  protocol=json.loads(read(study/'protocol.json'));progress=json.loads(read(study/'PROGRESS.json'))
  cp=root/protocol['source_census']['path'];census_bytes=read(cp);census=json.loads(census_bytes)
  require(digest(census_bytes)==protocol['source_census']['sha256'],'Protocol/census SHA mismatch')
  entries={x['benchmark']:x for x in census['rows']};names=sorted(set(entries)|{x['benchmark']for x in census.get('remaining_assessments',[])})
  require(census['native_names']==len(names)==537,'537 unique native names required')
  for b in names:
   p=study/'benchmarks'/(b+'.json');doc=json.loads(read(p));require(doc['benchmark']==b,'Ledger name mismatch');require(set(doc['models'])==set(MODELS),'Four model ledger slots required');ledgers[b]=doc
   accepted=doc.get('accepted_named_contract')
   if accepted:
    require(b in entries,'Accepted ledger has no census contract')
    for k in ['result_sha256','raw_config_sha256','configuration_id']:require(accepted[k]==entries[b][k],'Ledger/census contract join mismatch '+b+'/'+k)
   cdir=study/'selected_contracts'/b
   if (cdir/'result.json.gz').exists():
    rb=gz(cdir/'result.json.gz');cb=gz(cdir/'config.json.gz');require(b in entries,'Frozen contract has no census entry');require(digest(rb)==entries[b]['result_sha256']and digest(cb)==entries[b]['raw_config_sha256'],'Frozen contract/census mismatch '+b);contracts[b]={'result_sha256':digest(rb),'config_sha256':digest(cb)}
   for m in MODELS:
    state_snapshot.append({'benchmark':b,'model':m,'state':copy_json(doc['models'][m])})
  manifests=sorted((study/'profiles').glob('*/*/*/manifest.json'))
  for index,mp in enumerate(manifests):
   identity=rel(mp.parent);row={'profile_id':identity,'manifest':rel(mp),'benchmark':mp.parent.parent.parent.name,'model':mp.parent.parent.name,'manifest_sha256':None,'integrity_valid':False};files={};digests={};observations={}
   try:
    manifest_raw=read(mp);row['manifest_sha256']=digest(manifest_raw);manifest=json.loads(manifest_raw)
    for item in manifest['files']:
     name=item['original_name'];require(name not in digests,'Duplicate manifest filename');path=(root/item['path']).resolve();require(path.is_relative_to(mp.parent.resolve()),'Frozen member escapes its manifest directory');require(path.name==name+'.gz','Frozen member filename mismatch')
     packed_hash=sha_file(path);inputs[rel(path)]=packed_hash;require(packed_hash==item['compressed_sha256'],'Compressed SHA mismatch '+str(path));require(path.stat().st_size==item['compressed_bytes'],'Compressed length mismatch')
     h=hashlib.sha256();length=0;parts=[];keep=name in ['plan.json','profile_provenance.json','counter_snapshot_recovery.json','counter_derivation_recovery.json','parent_profile_provenance.json','build_outcome.json','assert_child_diagnostics.json','tensor_mode_decisions.json'];scan=name in ['execution.log','build.log','counters.csv'];tail='';offset=0
     with gzip.open(path,'rb')as f:
      for data in iter(lambda:f.read(4<<20),b''):
       h.update(data);length+=len(data)
       if keep:
        require(length<=64<<20,'JSON member exceeds explicit 64MiB parsing budget');parts.append(data)
       if scan:
        text=tail+data.decode('utf8',errors='replace')
        for key,rx in RX.items():
         if key in observations:continue
         hit=rx.search(text)
         if hit:observations[key]={'member':name,'member_uncompressed_sha256':item['uncompressed_sha256'],'excerpt':text[max(0,hit.start()-120):min(len(text),hit.end()+300)],'match_text':hit.group(0)[:1000],'interpretation':'Observed retained diagnostic only; not an independently established root cause or translation failure.'}
        tail=text[-4096:];offset+=len(data)
     require(h.hexdigest()==item['uncompressed_sha256'],'Uncompressed SHA mismatch '+str(path));require(length==item['uncompressed_bytes'],'Uncompressed length mismatch');digests[name]=h.hexdigest();integrity_bytes+=length
     if keep:files[name]=json.loads(b''.join(parts))
    pr=files['profile_provenance.json'];plan=files['plan.json'];b,m=plan['benchmark'],plan['model'];require(b in names and m in MODELS,'Unknown frozen assessment slot');require(b==row['benchmark']and m==row['model'],'Manifest directory/plan slot mismatch');require(digests['plan.json']==pr['plan_sha256'],'Plan/provenance SHA mismatch');require(pr['gpu_uuid']==UUID,'Frozen device UUID mismatch');require(b in contracts,'Attempt missing frozen selected contract');require(plan['reference_result_sha256']==contracts[b]['result_sha256']and plan['raw_config_sha256']==contracts[b]['config_sha256'],'Plan/frozen config/result join mismatch');
    if plan['configuration_id']!=entries[b]['configuration_id']:
     qualifications.append({'profile_id':identity,'kind':'descriptive_configuration_identifier_differs','archived_plan_configuration_id':plan['configuration_id'],'current_census_configuration_id':entries[b]['configuration_id'],'raw_config_sha256':plan['raw_config_sha256'],'reference_result_sha256':plan['reference_result_sha256'],'meaning':'Exact config/result bytes join; retain both labels without claiming a source/math change or rewriting historical labels.'})
    collector=digests.get('collector_source.py')
    if collector is None:
     source=source_snapshot(pr['collector_sha256']);require(source is not None,'Historical collector snapshot missing');collector=digest(source)
    require(collector==pr['collector_sha256'],'Collector SHA mismatch')
    for phase,member in [('profile','execution.log'),('build','build.log')]:
     record=pr.get(phase,{})
     if record.get('log_sha256'):require(digests.get(member)==record['log_sha256'],phase+' log/provenance mismatch')
    if 'counters.csv'in digests and pr.get('counters_sha256')!=digests['counters.csv']:
     recovery=files.get('counter_snapshot_recovery.json');require(recovery and digests.get('counter_snapshot_at_derivation.csv')==pr['counters_sha256']==recovery['recovered_snapshot_sha256']and digests['counters.csv']==recovery['later_counter_sha256'],'Counter SHA mismatch without exact saved snapshot recovery');require('derived.json'not in digests and pr['status']!='profile_process_completed','Late append recovery cannot certify completed derivation')
     prefix_bytes=0
     with gzip.open(mp.parent/'counter_snapshot_at_derivation.csv.gz','rb')as snapshot,gzip.open(mp.parent/'counters.csv.gz','rb')as later:
      for block in iter(lambda:snapshot.read(4<<20),b''):
       require(later.read(len(block))==block,'Recovered counter snapshot is not exact later-byte prefix');prefix_bytes+=len(block)
     require(prefix_bytes==recovery['recovered_snapshot_bytes'],'Recovered prefix byte count mismatch')
     qualifications.append({'profile_id':identity,'kind':'late_counter_append_exact_snapshot_recovery','note':'Original timed-out sample prefix recovery does not turn failed process into accepted counter coverage.'})
    cpu_recovery=pr.get('CPU_only_recovery')is True
    if cpu_recovery:
     recovery=files['counter_derivation_recovery.json'];parent=files['parent_profile_provenance.json'];require(digests['parent_profile_provenance.json']==pr['counter_derivation_recovery_from']['sha256']==recovery['parent']['sha256'],'CPU recovery parent SHA mismatch');require(digests['counter_derivation_recovery.json']==pr['counter_derivation_recovery_from']['receipt_sha256'],'CPU recovery receipt SHA mismatch');require(parent['profile']==pr['profile']and parent['profile']['returncode']==0 and parent['profile']['stop_reason']is None,'CPU recovery execution altered/unsuccessful');require(pr.get('new_gpu_processes')==0 and pr.get('new_build_processes')==0,'CPU recovery new process claim');require(pr['additional_current_profile_input_bindings']==parent['additional_current_profile_input_bindings'],'Recovery invented historical source bindings');require(digests['recovery_tool_source.py']==recovery['tool']['sha256'],'Recovery tool snapshot SHA mismatch');require(digests['counters.csv']==parent['counters_sha256']==recovery['raw_counter']['sha256'],'Recovered counter bytes differ from actual parent');require(digests['plan.json']==parent['plan_sha256'],'Recovered historical plan differs')
    state=ledgers[b]['models'][m];na=False;review=state.get('profile_applicability_review')
    if review and review.get('parent_provenance_sha256')==digests['profile_provenance.json']:
     rr=root/review['path'];data=read(rr);require(digest(data)==review['sha256'],'Applicability review SHA mismatch');na=True
    profile=pr.get('profile',{});build=pr.get('build',{});recorded_profile='returncode'in profile;recorded_build='returncode'in build
    row.update(benchmark=b,model=m,configuration_id=plan['configuration_id'],reference_result_sha256=plan['reference_result_sha256'],raw_config_sha256=plan['raw_config_sha256'],provenance_sha256=digests['profile_provenance.json'],captured_collector_sha256=pr['collector_sha256'],raw_status=pr['status'],started_at=pr.get('started_at'),finished_at=pr.get('finished_at'),raw_error=pr.get('error'),counter_parse_error=pr.get('counter_parse_error'),raw_profile_returncode=profile.get('returncode'),raw_profile_stop_reason=profile.get('stop_reason'),raw_profile_wall_seconds=profile.get('wall_seconds'),raw_build_returncode=build.get('returncode'),raw_build_stop_reason=build.get('stop_reason'),raw_build_wall_seconds=build.get('wall_seconds'),planned_build_command=bool(plan.get('build_command')),recorded_profile_launcher_process=recorded_profile and not cpu_recovery,recorded_build_launcher_process=recorded_build and not cpu_recovery,original_process_record_reused=cpu_recovery,recorded_new_profile_launcher_process_count=1 if recorded_profile and not cpu_recovery else 0,recorded_new_build_launcher_process_count=1 if recorded_build and not cpu_recovery else 0,actual_new_profile_process_total=0 if cpu_recovery else 1 if recorded_profile else None,actual_new_build_launcher_total=0 if cpu_recovery else 1 if recorded_build else None,process_units='Recorded profiler-command or build-command launcher invocation; may spawn multiple compiler/target/rank processes. Not device JIT counts. Underlying total child/compiler processes remain unknown.',observations=observations,source_reviewed_runtime_NA=na,source_applicability_review=review if na else None,recorded_kernel_count=pr.get('kernel_count'),source_binding_after_record=pr.get('binding_recheck_after'),translation_failure_inference=False,underlying_device_JIT_invocations=None,underlying_total_target_or_compiler_processes=None,manifest_member_hashes=digests)
    row['operational_category']=classify(pr['status'],pr,observations,na);row['integrity_valid']=True
   except Exception as error:
    row['integrity_error']=str(error);issues.append({'profile_id':identity,'error':str(error)})
   attempts.append(row)
   if (index+1)%250==0:print(json.dumps({'frozen_attempts_verified':index+1,'total':len(manifests),'issues':len(issues)}),file=sys.stderr,flush=True)
  # Join selected and latest pointers without choosing a new capture or rewriting history.
  byid={r['profile_id']:r for r in attempts};views=[]
  for rec in state_snapshot:
   b,m,state=rec['benchmark'],rec['model'],rec['state'];history=state.get('profile_attempts',[])
   for item in history:
    if item['profile_id']not in byid:issues.append({'benchmark':b,'model':m,'error':'Ledger attempt missing frozen manifest '+item['profile_id']})
   for view in ['selected','latest']:
    selected=state.get('profile_ids',[]);latest=state.get('latest_profile_attempt');ids=selected if view=='selected'else[latest['profile_id']]if latest else[];status=state['profiling_status']if view=='selected'else latest['status']if latest else state['profiling_status'];linked=[byid.get(i)for i in ids];linked=[x for x in linked if x];category=classify(status,{}, {},status.startswith('roofline_not_applicable'))
    if linked:
     chosen=linked[-1]
     if not chosen.get('integrity_valid'):category='evidence_integrity_unverified'
     elif view=='selected'and status.startswith('roofline_not_applicable'):category='source_reviewed_runtime_no_kernel_not_applicable'
     elif view=='selected'and status=='profile_completed_quality_unverified':category='completed_process_quality_unverified'
     else:category=chosen['operational_category']
     if view=='selected'and chosen.get('provenance_sha256')and state.get('profile_provenance_sha256')and state['profile_provenance_sha256']!=chosen.get('provenance_sha256'):issues.append({'benchmark':b,'model':m,'error':'Selected provenance SHA mismatch'})
    unfrozen=[]
    for identity in ids:
     if identity not in byid:
      if identity.startswith('/tmp/hecbench-roofline-v100-20261020/')and identity.endswith('/profile_provenance.json'):
       unfrozen.append(identity);category='pending_freeze_of_live_reported_attempt'
       qualifications.append({'benchmark':b,'model':m,'view':view,'kind':'live_raw_pointer_not_frozen_yet','path':identity,'reported_status':status,'meaning':'No mutable raw evidence read or promoted. Retain reported status separately; frozen-integrity proof pending. Latest view remains latest frozen attempt.'})
      else:issues.append({'benchmark':b,'model':m,'error':view+' pointer missing frozen manifest '+identity})
    views.append({'view':view,'benchmark':b,'model':m,'denominator_model_slots':537,'status':status,'operational_category':category,'profile_ids':ids,'evidence_integrity_valid':None if unfrozen else all(x.get('integrity_valid')for x in linked)if ids else None,'unfrozen_live_pointer_pending':unfrozen,'latest_scope':'latest frozen finished attempt; mutable active raw directories are not imported','recorded_profile_returncodes':[x.get('raw_profile_returncode')for x in linked],'recorded_build_returncodes':[x.get('raw_build_returncode')for x in linked],'pending_or_no_frozen_attempt':not ids or bool(unfrozen),'plan_status':state.get('profile_plan_status'),'plan_blocking_gates':state.get('profile_plan_blocking_gates',[]),'primary_numerical_outcome':(ledgers[b].get('accepted_named_contract')or{}).get('outcome'),'profiling_is_not_translation_failure':True})
  require(len(views)==537*4*2,'Full selected/latest denominator failure')
  ledger_drift=[p for p,h in inputs.items()if '/benchmarks/'in p and sha_file(root/p)!=h]
  if ledger_drift:qualifications.append({'kind':'live_ledger_changed_despite_shared_coordinator_lock','paths':ledger_drift,'meaning':'A noncooperating live writer changed ledger during read. Individual consumed bytes remain SHA-bound; global coherent ledger snapshot not established. Frozen evidence checks remain independent.'})
  unlocked_at=stamp();fcntl.flock(lock,fcntl.LOCK_UN)
 out.mkdir(parents=True);summary=[];model_census=[];process_counts=collections.Counter()
 for view in ['selected','latest','all_frozen_attempts']:
  for model in MODELS:
   rr=[r for r in (attempts if view=='all_frozen_attempts'else views)if r.get('model')==model and(view=='all_frozen_attempts'or r['view']==view)]
   for category,count in sorted(collections.Counter(r.get('operational_category','evidence_integrity_unverified')for r in rr).items()):summary.append({'view':view,'model':model,'category':category,'count':count,'denominator_model_slots':537,'records_in_view':len(rr),'record_unit':'frozen_attempt_record'if view=='all_frozen_attempts'else'benchmark_model_slot','percent':None})
 for m in MODELS:
  ar=[x for x in attempts if x.get('model')==m];sr=[x for x in views if x['model']==m and x['view']=='selected'];model_census.append({'model':m,'benchmark_denominator':537,'selected_slots':len(sr),'latest_slots':sum(x['model']==m and x['view']=='latest'for x in views),'slots_with_frozen_attempt':len({x['benchmark']for x in ar}),'slots_without_frozen_attempt':537-len({x['benchmark']for x in ar}),'all_frozen_attempt_records':len(ar),'new_recorded_profile_launcher_processes':sum(x.get('recorded_new_profile_launcher_process_count',0)for x in ar),'new_recorded_build_launcher_processes':sum(x.get('recorded_new_build_launcher_process_count',0)for x in ar),'CPU_recovery_records_no_new_execution':sum(x.get('original_process_record_reused',False)for x in ar),'selected_status_counts':countdict(x['status']for x in sr),'selected_category_counts':countdict(x['operational_category']for x in sr),'underlying_device_JIT_invocations':None})
  for x in ar:
   if x.get('recorded_profile_launcher_process'):process_counts[(m,'profile_command_launcher',str(x.get('raw_profile_returncode')),str(x.get('raw_profile_stop_reason')))]+=1
   if x.get('recorded_build_launcher_process'):process_counts[(m,'build_command_launcher',str(x.get('raw_build_returncode')),str(x.get('raw_build_stop_reason')))]+=1
 usage=None
 if a.usage_receipt:usage={'path':str(a.usage_receipt.resolve()),'sha256':sha_file(a.usage_receipt)}
 bindings=dict(sorted(inputs.items()));payload={'schema_version':1,'created_at':now,'campaign_status':'incomplete_live_snapshot','source_progress_updated_at':progress.get('updated_at'),'snapshot_consistency':'coordinator LOCK_SH across consumed metadata; post-read ledger hash check '+('detected noncooperating writer drift'if ledger_drift else'unchanged in observed interval'),'live_ledger_drift_paths':ledger_drift,'lock':str(a.lock_path),'lock_acquired_at':locked_at,'lock_released_at':unlocked_at,'denominators':{'benchmarks':537,'models':list(MODELS),'model_slots':2148,'each_model_denominator':537,'selected_and_latest_are_separate_views_not_added':True,'all_frozen_attempts_are_not_unique_benchmark_counts':True},'per_model':model_census,'summary_rows':summary,'integrity':{'valid':not issues,'issues':issues,'qualifications':qualifications,'frozen_attempt_records':len(attempts),'all_manifest_uncompressed_bytes_verified':integrity_bytes,'level':'Every compressed/raw memberSHA+length, capturedplan/provenance/collector/log+archivedselectedconfig/result joins. No numerical counter rederivation, GPU recertification or today-source historical backfill.'},'operational_process_units':'Recorded profiler/build launcher processes, not underlying child processes or compiler/deviceJIT. CPU-only derivation recovery copies do not add executions. Unknowns null, never zero. No translation failures inferred.','classification_method':'Stop reasons and observed retained log signatures; log categories are evidence observations, not unique causal diagnoses. Pending/hardware/replay/NA remain visible.','input_bindings':bindings,'input_digest':digest(json.dumps(bindings,sort_keys=True,separators=(',',':')).encode()),'exporter_sha256':sha_file(Path(__file__)),'shared_measurement_usage_receipt':usage}
 (out/'exporter_source.py').write_bytes(Path(__file__).read_bytes())
 compact={k:payload[k]for k in ['schema_version','created_at','campaign_status','source_progress_updated_at','snapshot_consistency','denominators','per_model','summary_rows','operational_process_units','classification_method','input_digest','exporter_sha256','shared_measurement_usage_receipt']}
 compact['full_evidence_summary']='summary.json';compact['full_evidence_summary_SHA_note']='Full summary carries all exact input hashes; compact counts are not an independent raw recertification.';compact['integrity']={k:payload['integrity'][k]for k in ['valid','frozen_attempt_records','all_manifest_uncompressed_bytes_verified','level']};compact['integrity']['issue_count']=len(issues);compact['integrity']['qualification_count']=len(qualifications)
 dump(out/'manuscript_summary.json',compact)
 dump(out/'summary.json',payload);dump(out/'all_frozen_attempts.json',{'records':attempts});dump(out/'selected_latest_slots.json',{'records':views});dump(out/'integrity.json',payload['integrity'])
 write_csv(out/'operational_summary.csv',summary)
 write_csv(out/'model_denominators.csv',[{k:json.dumps(v,sort_keys=True)if isinstance(v,dict)else v for k,v in row.items()}for row in model_census])
 write_csv(out/'process_returncodes.csv',[{'model':m,'recorded_process_unit':kind,'returncode':rc,'stop_reason':stop,'count':n}for(m,kind,rc,stop),n in sorted(process_counts.items())])
 write_csv(out/'attempts.csv',[{k:json.dumps(v,sort_keys=True)if isinstance(v,(list,dict))else v for k,v in x.items()if k not in ['observations','manifest_member_hashes','process_units']}for x in attempts])
 write_csv(out/'selected_latest.csv',[{k:json.dumps(v,sort_keys=True)if isinstance(v,(list,dict))else v for k,v in x.items()}for x in views])
 lines=['# V100 measurement operational-outcome census','',f'Snapshot: {progress.get("updated_at")}. Integrity issues: {len(issues)}. All four models retain denominator537; selected and latest outcomes are separate from repeated attempts. No percentages suppress pending or blocked slots.','', '| Model | Selected completed counters | Runtime NA | Pending/active | Hardware | Replay holds | Plan gates | Other selected | Frozen attempts | Profile launchers | Build launchers |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
 for x in model_census:
  c=x['selected_category_counts'];keys=['completed_counter_observation','source_reviewed_runtime_no_kernel_not_applicable','pending_or_active_not_completed','native_hardware_barrier','replay_method_hold','plan_isolation_or_device_gate'];v=[c.get(k,0)for k in keys];v[2]+=c.get('pending_freeze_of_live_reported_attempt',0);lines.append('| '+x['model']+' | '+' | '.join(map(str,v+[537-sum(v),x['all_frozen_attempt_records'],x['new_recorded_profile_launcher_processes'],x['new_recorded_build_launcher_processes']]))+' |')
 lines+=['','Operational failures are profiling/build/runtime/isolation/dependency/parse observations, not translation failures. Cause remains unknown when only a generic nonzero exit is recorded. Profile/build launcher counts may include nested target/compiler/rank subprocesses; total backend compiler/JIT counts remain unknown. CPU-only recovery records add no execution. Runtime allocation/copy NA does not manufacture zero FLOPs or bytes. Successful kernel samples do not prove full ROI or causal bottlenecks.','', 'Exact parent evidence and diagnostic snippets: all_frozen_attempts.json. Complete537×4 selected/latest rows: selected_latest_slots.json. All compressed/raw input SHAs and consistency scope: summary.json. Historical executed source identity is retained from captured provenance; current source files are never rehashed to certify past runs.','','Run this exporter again to a new /tmp directory after future metadata updates; existing output directories are refused. No GPU, benchmark tests, compiler or target program is executed by this exporter.']
 (out/'README.md').write_text('\n'.join(lines)+'\n');print(json.dumps({'output':str(out),'attempts':len(attempts),'model_slots_per_view':2148,'integrity_issues':len(issues)}));return 0 if not issues else 1

def copy_json(x):return json.loads(json.dumps(x))
def write_csv(p,rows):
 keys=list(dict.fromkeys(k for row in rows for k in row))
 with Path(p).open('w',newline='')as f:
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
if __name__=='__main__':raise SystemExit(main())
