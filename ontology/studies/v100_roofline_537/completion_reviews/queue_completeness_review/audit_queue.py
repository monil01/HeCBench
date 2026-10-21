from pathlib import Path
import json,hashlib,os,datetime,fcntl,collections,gzip,csv,io
R=Path('/home/imo/HeCBench_julia');S=R/'ontology/studies/v100_roofline_537';RAW=Path('/tmp/hecbench-roofline-v100-20261020');O=Path(__file__).resolve().parent;D=S/'completion_reviews/completion_supervisor'
def sha(b):return hashlib.sha256(b).hexdigest()
inputs={}
def read(p):
 p=Path(p);b=p.read_bytes();inputs[str(p)]={'sha256':sha(b),'bytes':len(b)};return b
def j(p):return json.loads(read(p))
def values(args,key):return [args[i+1]for i,x in enumerate(args[:-1])if x==key]
def proc(pid,start=None):
 p=Path('/proc')/str(pid)
 try:
  st=(p/'stat').read_text().rsplit(')',1)[1].split();raw=(p/'cmdline').read_bytes();a=[x.decode()for x in raw.split(b'\0')if x]
  return {'pid':pid,'start_ticks':st[19],'state':st[0],'uid':p.stat().st_uid,'command':a,'command_sha256':sha(raw),'cwd':str((p/'cwd').resolve()),'children':(p/'task'/str(pid)/'children').read_text().split(),'matches_expected_generation':start is None or str(start)==st[19],'live_expected_owned':(start is None or str(start)==st[19])and st[0]not in ['Z','X']and p.stat().st_uid==os.getuid()}
 except(FileNotFoundError,ProcessLookupError,PermissionError):return {'pid':pid,'expected_start_ticks':start,'live_expected_owned':False,'status':'absent_or_unreadable'}
def rootpath(value,cwd):
 p=Path(value);return p.resolve()if p.is_absolute()else(Path(cwd)/p).resolve()
started=datetime.datetime.now(datetime.timezone.utc).isoformat()
with Path('/tmp/hecbench-roofline-metadata.lock').open('r')as lock:
 fcntl.flock(lock,fcntl.LOCK_SH)
 ledgers={p.stem:j(p)for p in sorted((S/'benchmarks').glob('*.json'))}
 adoption=j(D/'root_adoption_command.json'); config=j(RAW/'completion_20261021_v12/configuration.json'); checkpoint=j(RAW/'completion_20261021_v12/checkpoint.json'); custom=j(D/'controller_reviews_v12.json')
 census=j(R/'src/_portable_study/completion537_20261020/current537_census.json')
 controller=[proc(x['pid'],x['start_ticks'])for x in config['controllers']];supervisor=proc(adoption['pid'],adoption['start_ticks'])
 descendant_collectors=[]
 for c in controller:
  if not c['live_expected_owned']:continue
  todo=list(c.get('children',[]));seen=set()
  while todo:
   pid=int(todo.pop())
   if pid in seen:continue
   seen.add(pid);child=proc(pid)
   if not child['live_expected_owned']:continue
   todo.extend(child.get('children',[]))
   if any(Path(x).name=='v100_roofline_collect.py'for x in child.get('command',[])):
    child['monitored_ancestor_identity']=str(c['pid'])+':'+c['start_ticks'];descendant_collectors.append(child)

 planrecords=[];badplans=[]
 for pp in sorted(RAW.glob('batch*/*/*/plan.json')):
  try:
   b=read(pp);d=json.loads(b);pr=pp.parent/'profile_provenance.json';provenance=json.loads(read(pr))if pr.is_file()else None
   planrecords.append({'path':str(pp),'sha256':sha(b),'batch_root':str(pp.parents[2]),'benchmark':d['benchmark'],'model':d['model'],'configuration_id':d.get('configuration_id'),'staged':d.get('staged'),'blocking_gates':d.get('blocking_gates'),'provenance_path':str(pr)if pr.is_file()else None,'raw_status':provenance.get('status')if provenance else'plan_only_unstarted','raw_stop_reason':(provenance.get('profile')or{}).get('stop_reason')if provenance else None,'raw_profile_started':bool(provenance and provenance.get('profile')),'metadata_only_plan':d.get('staged')is not True,'plan':d,'provenance':provenance})
  except(KeyError,ValueError,OSError)as e:badplans.append({'path':str(pp),'error':repr(e)})
 journalpath=RAW/'campaign_003.jsonl';journal=[]
 if journalpath.exists():
  for line in read(journalpath).decode().splitlines():
   try:journal.append(json.loads(line))
   except ValueError:pass
 frozen={e['benchmark']for e in journal if e.get('event')=='frozen'}
 followupjournals={}
 for c in controller:
  if not c['live_expected_owned']:continue
  for q in values(c['command'],'--journal'):
   p=rootpath(q,c['cwd']);rows=[]
   if p.is_file():
    for line in read(p).decode().splitlines():
     try:rows.append(json.loads(line))
     except ValueError:pass
   followupjournals[str(p)]=rows
 selected={}
 for b,d in ledgers.items():
  for m,s in d['models'].items():
   z=None
   mp=R/s['profile_evidence_manifest']if s.get('profile_evidence_manifest')else None
   if mp and mp.is_file():
    manifest=j(mp);members={}
    for name in ['plan.json','profile_provenance.json']:
     a=[x for x in manifest.get('files',[])if x.get('original_name')==name]
     if len(a)==1:
      f=R/a[0]['path'];packed=read(f);raw=gzip.decompress(packed)
      if sha(packed)!=a[0]['compressed_sha256']or sha(raw)!=a[0]['uncompressed_sha256']:raise RuntimeError('Frozen selected binding mismatch '+str(f))
      members[name]=json.loads(raw)
    if len(members)==2:z=members
   selected[(b,m)]={'state':s,'frozen_selected':z}
 frozen_histories={}
 for key,z in selected.items():
  if z['state'].get('profiling_status')!='profile_process_failed':continue
  histories=[]
  for slot in sorted((S/'profiles'/key[0]/key[1]).glob('*')):
   if not(slot/'manifest.json').is_file():continue
   manifest=j(slot/'manifest.json');members={}
   for name in ['plan.json','profile_provenance.json']:
    items=[x for x in manifest['files']if x['original_name']==name]
    if len(items)!=1:continue
    entry=items[0];packed=read(R/entry['path']);raw=gzip.decompress(packed)
    if sha(packed)!=entry['compressed_sha256']or sha(raw)!=entry['uncompressed_sha256']:raise RuntimeError('History SHA drift')
    members[name]=json.loads(raw)
   if len(members)==2:histories.append({'profile_id':str(slot.relative_to(R)),'members':members})
  frozen_histories[key]=histories
 snapshot_end=datetime.datetime.now(datetime.timezone.utc).isoformat()
# Source semantics copied descriptively, never imported from the target tools.
STANDARD=set('benchmark model configuration_id reference_result reference_result_sha256 raw_config_sha256 effective_config_adjudication outcome qualification source_cwd_readonly_inputs scratch_cwd build_command profile_target_command environment_overrides runtime_assets runtime_asset_bytes execution_bindings source_path_literal_findings review blocking_gates notes staging_status effective_original_timing_region kernel_ROI_mapping_status profiler_requirements commands_executed binding_validation staged'.split())
METHOD_FIELDS=('application_replay','profile_window','isolated_assertion','large_artifact_policy','tensor_mode_bindings_before_profile','tensor_mode_decisions_sha256','selection_override')
def normalized(v,p):
 if isinstance(v,str):return v.replace(p['scratch_cwd'],'<FRESH_SCRATCH>')
 if isinstance(v,list):return[normalized(x,p)for x in v]
 if isinstance(v,dict):return{normalized(k,p):normalized(x,p)for k,x in v.items()}
 return v
def signature(p):return{k:normalized(p.get(k),p)for k in ['build_command','profile_target_command','environment_overrides','source_cwd_readonly_inputs','effective_original_timing_region','reference_result_sha256','raw_config_sha256','outcome']}
def holds(p,v):
 out=['noncanonical_plan_field:'+k for k in sorted(set(p)-STANDARD)]
 out+=['custom_collector_method:'+k for k in METHOD_FIELDS if v.get(k)]
 if p['benchmark']in ['perlin','zmddft','su3']:out+=['large_payload_optin_required']
 if p['benchmark']in ['assert','atomicSystemWide','streamUM','rayleighBenardConvection']:out+=['explicit_case_method_review_required']
 return out
routes=collections.defaultdict(list);controller_summary=[]
for c in controller+descendant_collectors:
 a=c.get('command',[])
 if not c['live_expected_owned']:controller_summary.append(c);continue
 roots=values(a,'--plan-root');kind='direct_or_followup'
 # Private one-shot follower links collector argv through its exact launch review.
 if any(Path(x).name=='wait_then_collect.py'for x in a):
  script=next(Path(x)for x in a if Path(x).name=='wait_then_collect.py');lp=script.parent/'launch_review.json';lr=j(lp)
  if values(a,'--review-sha256')==[sha(lp.read_bytes())]:roots=values(lr['collector_argv'],'--plan-root');kind='reviewed_private_follower'
  else:kind='private_follower_review_mismatch';roots=[]
 if any(Path(x).name=='v100_roofline_campaign.py'for x in a):
  kind='main_campaign';excluded=set(values(a,'--exclude-benchmark'));allowed=set(values(a,'--benchmark'));names={x['benchmark']for x in census['rows']};remaining=sorted(b for b in names if b not in excluded and b not in frozen and(not allowed or b in allowed))
  stop=values(a,'--stop-file');stop_exists=bool(stop and rootpath(stop[0],c['cwd']).exists());c.update(excluded_benchmarks=sorted(excluded),remaining_unfrozen_names=remaining,stop_marker_present=stop_exists)
  if not stop_exists:
   for b in remaining:
    for m in ['cuda','cuda_jl','ka','jacc']:
     oldplans=[p for p in planrecords if p['batch_root']==str(RAW/('batch003-'+b))and p['model']==m]
     if oldplans and oldplans[0]['raw_status']not in ['plan_only_unstarted','started','waiting_for_idle_gpu']:continue
     routes[(b,m)].append({'identity':str(c['pid'])+':'+c['start_ticks'],'kind':'main_campaign_not_yet_frozen','qualification':'Generates or traverses selected name once; existing failed raw attempts are preserved, not retried.'})
 processed=set()
 for jp in values(a,'--journal'):
  processed|={e['plan_root']for e in followupjournals.get(str(rootpath(jp,c['cwd'])),[])if e.get('event')=='followup_collect'}
 for r in roots:
  root=str(rootpath(r,c['cwd']))
  if root in processed:continue
  for p in planrecords:
   if p['batch_root']!=root or p['benchmark']in values(a,'--exclude-benchmark'):continue
   # Existing terminal evidence is never reexecuted by the collector.
   if p['raw_status']not in ['plan_only_unstarted','started','waiting_for_idle_gpu']:continue
   isolated=(p['benchmark']=='assert'and '--isolated-assertion-reviewed'in a and p['blocking_gates']==['requires_isolated_assertion_collector_phase_hooks'])
   if (p['blocking_gates']or not p['staged'])and not isolated:continue
   routes[(p['benchmark'],p['model'])].append({'identity':str(c['pid'])+':'+c['start_ticks'],'kind':kind,'plan':p['path'],'raw_status':p['raw_status'],'qualification':'Live controller exact owned generation and command. Future success not guaranteed; collector may skip newer selected success.'})
 c['coverage_kind']=kind;controller_summary.append(c)
rows=[];counts=collections.Counter();uncovered=[];operational=[]
for (b,m),z in sorted(selected.items()):
 s=z['state'];status=s.get('profiling_status');f=z['frozen_selected'];reason=((f or{}).get('profile_provenance.json',{}).get('profile')or{}).get('stop_reason'); rr=routes[(b,m)];mh=[]
 # Historical raw custom-method fields are preserved as conservative retry holds.
 histories=[p for p in planrecords if p['benchmark']==b and p['model']==m and p['provenance']]
 for hrow in frozen_histories.get((b,m),[]):
  hp=hrow['members']['plan.json'];hv=hrow['members']['profile_provenance.json'];h=holds(hp,hv)
  if f and signature(hp)!=signature(f['plan.json']):h.append('historical_execution_method_signature_differs_from_selected')
  dl=hv.get('operational_deadline')or{}
  if hv.get('profile')and not dl:h.append('historical_deadline_missing_no_invented_bound')
  if any(isinstance(dl.get(k),(int,float))and dl[k]>3600 for k in ['requested_seconds','effective_seconds']):h.append('historical_deadline_exceeds_explicit3600_cap_requires_review')
  if h:mh.append({'profile_id':hrow['profile_id'],'holds':h})
 if status=='profile_process_completed':category='selected_completed_observation_not_complete_ROI'
 elif status=='native_hardware_blocked':category='native_hardware_barrier'
 elif b in ['p2p','pingpong','ring']:category='transport_scalar_not_applicable_requires_separate_communication_tier'
 elif b=='allreduce':category='two_GPU_permission_or_reviewed_local_rank_method_gate'
 elif b=='ccl':category='two_GPU_vendor_collective_method_gate'
 elif rr:category='live_controller_route_present_not_success_claim'
 elif status=='profile_process_failed'and reason in ['external_gpu_activity_detected_own_profile_stopped','wall_timeout']and not mh and supervisor['live_expected_owned']:
  category='finite039_041_conditional_generic_retry_candidate';operational.append({'benchmark':b,'model':m,'reason':reason})
 elif status=='profile_process_failed'and reason in ['external_gpu_activity_detected_own_profile_stopped','wall_timeout']:
  category='uncovered_custom_or_method_held_operational_failure'
 elif status in ['roofline_not_applicable_no_device_kernels','source_not_applicable','no_device_kernel_work']:
  category='source_no_device_work_applicability'
 else:category='uncovered_repairable_or_unresolved_requires_root_adjudication'
 plans=[{k:p[k]for k in ['path','sha256','batch_root','configuration_id','staged','blocking_gates','raw_status','raw_stop_reason']}for p in planrecords if p['benchmark']==b and p['model']==m]
 row={'benchmark':b,'model':m,'selected_status':status,'selected_profile_ids':s.get('profile_ids',[]),'selected_evidence_manifest':s.get('profile_evidence_manifest'),'scalar_roofline_applicability':s.get('scalar_roofline_applicability'),'selected_stop_reason':reason,'coverage_category':category,'live_routes':rr,'historical_custom_method_holds':mh,'raw_plans':plans,'selected_ROI_coverage':s.get('roi_coverage'),'whole_ROI_claim':False,'causal_claim':False}
 rows.append(row);counts[category]+=1
 if category.startswith('uncovered_'):uncovered.append(row)
assert len(ledgers)==537 and len(rows)==2148
for p in ['tools/v100_roofline_completion.py','tools/v100_roofline_retry.py','tools/v100_roofline_campaign.py','tools/v100_roofline_followups.py','tools/v100_roofline_handoff.py']:read(R/p)
report={'schema_version':1,'snapshot_started_at':started,'selected_metadata_snapshot_finished_at':snapshot_end,'denominators':{'names':537,'models':4,'slots':2148},'source_snapshot_policy':'Selected ledgers,selected frozen manifest/plan/provenance,raw plans/provenance and controller receipts read under existing LOCK_SH. Process identities are point-in-time /proc observations; no GPU queries. Raw active journals/provenance may progress later.','selected_status_counts':dict(collections.Counter(r['selected_status']for r in rows)),'coverage_category_counts':dict(counts),'supervisor':supervisor,'supervisor_checkpoint_status':checkpoint.get('status'),'finite_retry_rounds':config['retry_batch_ids'],'finite_retry_rounds_already_consumed':checkpoint.get('rounds'),'controller_receipt_identities':len(controller),'actual_live_expected_owned_controllers':sum(c['live_expected_owned']for c in controller),'controller_frontier':controller_summary,'raw_plan_count':len(planrecords),'raw_plan_read_errors':badplans,'uncovered_slot_count':len(uncovered),'uncovered_slots':uncovered,'slots':rows,'input_bindings':inputs,'qualifications':['A live plan/controller route is not execution, numeric agreement, complete ROI or causal proof.','Finite039/041 admit only selected timeout/foreignactivity failures; historical custom fields/signature/deadline or newly prepared plan/binding inequivalence can hold execution. Conditional generic candidates are not guaranteed queued GPU executions.','Raw plans with terminal provenance are preserved, not automatically reexecuted by a live parent; unstarted plans without actual controller coverage are not queued.','Transport scalarNA12,hardware20,allreduce permission/method4,CCL method4 remain separate if their selected state changes. Categories report actual current ledger first, no forced cardinality.','No arbitrary future frontiers or failed raw evidence removed. Root must re-evaluate uncovered slots after current controllers and finite retries; no launch performed by this audit.'],'GPU_queries':0,'GPU_runs':0,'shared_edits':False}
(O/'queue_coverage.json').write_text(json.dumps(report,indent=2)+'\n')
with(O/'slots.csv').open('w',newline='')as f:
 w=csv.DictWriter(f,fieldnames=['benchmark','model','selected_status','coverage_category','selected_stop_reason','live_controller_count','raw_plan_count']);w.writeheader();w.writerows({**{k:r[k]for k in ['benchmark','model','selected_status','coverage_category','selected_stop_reason']},'live_controller_count':len(r['live_routes']),'raw_plan_count':len(r['raw_plans'])}for r in rows)
print(json.dumps({k:report[k]for k in ['selected_status_counts','coverage_category_counts','controller_receipt_identities','actual_live_expected_owned_controllers','raw_plan_count','uncovered_slot_count']}));print('uncovered',[(r['benchmark'],r['model'],r['selected_status'],r['selected_stop_reason'])for r in uncovered])
