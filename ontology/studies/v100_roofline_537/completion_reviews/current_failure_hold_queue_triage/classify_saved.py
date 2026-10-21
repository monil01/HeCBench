from pathlib import Path
import json,gzip,hashlib,collections
D=Path(__file__).resolve().parent;R=Path('/home/imo/HeCBench_julia');v=json.loads(gzip.decompress((D/'ledger_snapshot.json.gz').read_bytes()));rows=[];errors=[]
def sha(b):return hashlib.sha256(b).hexdigest()
def member(folder,manifest,name):
 es=[x for x in manifest['files']if x['original_name']==name]
 if not es:return None,None
 if len(es)!=1:raise ValueError('Duplicate member')
 e=es[0];p=R/e['path'];b=p.read_bytes();raw=gzip.decompress(b)
 if sha(b)!=e['compressed_sha256']or sha(raw)!=e['uncompressed_sha256']or len(raw)!=e['uncompressed_bytes']:raise ValueError('Member identity drift')
 return raw,{'path':str(p),'compressed_sha256':sha(b),'raw_sha256':sha(raw),'raw_bytes':len(raw)}
def captured(b,m,pid,view):
 folder=R/pid;mp=folder/'manifest.json';mb=mp.read_bytes();manifest=json.loads(mb);pr,pref=member(folder,manifest,'profile_provenance.json');pp,planref=member(folder,manifest,'plan.json');prov=json.loads(pr);plan=json.loads(pp)
 if prov['plan_sha256']!=sha(pp):raise ValueError('Plan/provjoin')
 stop=(prov.get('profile')or{}).get('stop_reason');category='operational_'+stop if stop else 'unknown_runtime_or_collection_cause';profile=prov.get('profile')or{};build=prov.get('build')or{};diagnostics=[];logrefs=[]
 for n in ['execution.log','build.log','counters.csv']:
  raw,ref=member(folder,manifest,n)
  if raw is None:continue
  logrefs.append(ref)
  # Preserve exact rawfilehash; bounded lines only for descriptive evidence.
  lines=raw.decode(errors='replace').splitlines()
  for line in lines:
   low=line.lower()
   if any(s in low for s in ['==error==','error:','fatal error','assertion','error: loaderror','not found','no such file','permission denied','no kernels were profiled','no child processes','glibcxx','safe_close','keyerror','runtimeerror','traceback','metric unit mismatch','invalid argument','killed','target process exited']):diagnostics.append(line[:1200])
  diagnostics=diagnostics[:16]
 if build and build.get('returncode')not in [None,0]:category='observed_build_process_failure'
 elif not stop and any('field larger than field limit'in x.lower()for x in diagnostics):category='observed_CSV_field_limit'
 elif not stop and any('safe_close'in x for x in diagnostics):category='observed_host_safe_close_abort_cause_unknown'
 elif not stop and any('glibcxx'in x.lower()for x in diagnostics):category='observed_dynamic_library_symbol_error'
 elif not stop and any('permission denied'in x.lower()for x in diagnostics):category='observed_permission_denied_guard_or_path_requires_source_review'
 elif not stop and any('no such file'in x.lower()or'not found'in x.lower()for x in diagnostics):category='observed_missing_path_or_dependency_requires_source_review'
 extra=sorted(set(plan)-set('benchmark model configuration_id reference_result reference_result_sha256 raw_config_sha256 effective_config_adjudication outcome qualification source_cwd_readonly_inputs scratch_cwd build_command profile_target_command environment_overrides runtime_assets runtime_asset_bytes execution_bindings source_path_literal_findings review blocking_gates notes staging_status effective_original_timing_region kernel_ROI_mapping_status profiler_requirements commands_executed binding_validation staged'.split()))
 row={'benchmark':b,'model':m,'view':view,'profile_id':pid,'manifest':{'path':str(mp),'sha256':sha(mb)},'provenance':pref,'plan':planref,'status':prov.get('status'),'stop_reason':stop,'profile_returncode':profile.get('returncode'),'build_returncode':build.get('returncode'),'category':category,'diagnostic_lines':diagnostics,'log_evidence':logrefs,'generic_retry_status_eligible':view=='selected'and prov.get('status')=='profile_process_failed'and stop in ['external_gpu_activity_detected_own_profile_stopped','wall_timeout'],'custom_plan_fields':extra,'custom_methods':[k for k in ['application_replay','profile_window','isolated_assertion','large_artifact_policy','selection_override','profiler_method']if prov.get(k)],'configuration_id':plan.get('configuration_id')}
 rows.append(row)
for b,ledger in v.items():
 for m,state in ledger['models'].items():
  selected=state['profiling_status'];latest=state.get('latest_profile_attempt')or{};pid=latest.get('profile_id')
  if selected in ['profile_process_failed','collection_failed']:
   ids=state.get('profile_ids')or[]
   if ids:
    try:captured(b,m,ids[0],'selected')
    except Exception as e:errors.append({'benchmark':b,'model':m,'view':'selected','error':repr(e)})
  if latest.get('status')in ['profile_process_failed','collection_failed','profile_completed_counter_derivation_failed']and pid and pid not in (state.get('profile_ids')or[])[:1]:
   try:captured(b,m,pid,'latest_not_selected')
   except Exception as e:errors.append({'benchmark':b,'model':m,'view':'latest','error':repr(e)})
(D/'frozen_failure_rows.json').write_text(json.dumps({'rows':rows,'integrity_issues':errors},indent=2)+'\n');print('selectedcategories',dict(collections.Counter(x['category']for x in rows if x['view']=='selected')));print('latestnotselected',len([x for x in rows if x['view']=='latest_not_selected']),'integrityissues',errors)
for x in rows:
 if x['view']=='selected'and x['category']not in ['operational_external_gpu_activity_detected_own_profile_stopped','operational_wall_timeout']:print(x['benchmark'],x['model'],x['category'],x['diagnostic_lines'][:3])
