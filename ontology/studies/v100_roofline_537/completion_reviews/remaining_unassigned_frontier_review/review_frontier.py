"""Bounded CPU-only snapshot: current blocked slots vs exact live finite plans."""
import os,json,hashlib,datetime,collections
from pathlib import Path
HERE=Path(__file__).resolve().parent;W=Path('/home/imo/HeCBench_julia');RAW=Path('/tmp/hecbench-roofline-v100-20261020');STUDY=W/'ontology/studies/v100_roofline_537';BASE=Path('/tmp/roofline-method-review/reviewed_plan_v4_plus_v5_cap512');STATUSES={'not_started_plan_blocked','not_started_replay_method_blocked','replay_method_blocked'}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(4<<20),b''):h.update(b)
 return h.hexdigest()
def meta(p):p=Path(p);return {'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size}
def dump(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def options(args,name):return [args[i+1]for i,x in enumerate(args[:-1])if x==name]
def snapshot_processes():
 out=[];allowed={'v100_roofline_collect.py','v100_roofline_followups.py','v100_roofline_campaign.py','v100_roofline_retry.py','v100_roofline_completion.py','v100_roofline_handoff.py'}
 for p in Path('/proc').iterdir():
  if not p.name.isdigit():continue
  try:
   if p.stat().st_uid!=os.getuid():continue
   args=[v.decode(errors='replace')for v in(p/'cmdline').read_bytes().split(b'\0')if v]
   scripts=[v for v in args[:3]if Path(v).name in allowed]
   if not scripts:continue
   st=(p/'stat').read_text().rsplit(')',1)[1].split()
   if st[0]=='Z':continue
   cwd=os.readlink(p/'cwd');script=Path(scripts[0]);script=script if script.is_absolute()else Path(cwd)/script
   rec={'pid':int(p.name),'uid':p.stat().st_uid,'ppid':int(st[1]),'sid':int(st[3]),'start_ticks':st[19],'state':st[0],'argv':args,'cwd':cwd,'controller_identity':p.name+':'+st[19],'script':str(script),'current_script_file':meta(script)if script.is_file()else None,'current_script_hash_is_not_proof_of_original_loaded_bytes':True,'plan_roots':options(args,'--plan-root'),'includes':options(args,'--benchmark'),'excludes':options(args,'--exclude-benchmark'),'wchan':(p/'wchan').read_text().strip()}
   out.append(rec)
  except(OSError,ValueError):pass
 return out
observed=datetime.datetime.now(datetime.timezone.utc).isoformat();processes=snapshot_processes();dump(HERE/'process_snapshot.json',{'observed_at':observed,'own_uid_only':os.getuid(),'processes':processes,'no_GPU_queries':True})
ledgers=[];targets=[];hardware=[];statuscounts=collections.Counter();ledger_snapshots=HERE/'ledger_snapshots';ledger_snapshots.mkdir(exist_ok=True);ledger_meta={}
for p in sorted((STUDY/'benchmarks').glob('*.json')):
 raw=p.read_bytes();d=json.loads(raw);digest=hashlib.sha256(raw).hexdigest();ledger_meta[str(p)]={'path':str(p),'sha256':digest,'bytes':len(raw)}
 if any(row.get('profiling_status')in STATUSES or row.get('profiling_status')=='native_hardware_blocked'for row in d.get('models',{}).values()):(ledger_snapshots/p.name).write_bytes(raw)
 for model,row in d.get('models',{}).items():
  statuscounts[row.get('profiling_status')]+=1
  if row.get('profiling_status')in STATUSES:targets.append((p,p.stem,model,row));ledgers.append(ledger_meta[str(p)])
  if row.get('profiling_status')=='native_hardware_blocked':hardware.append({'benchmark':p.stem,'model':model,'ledger':ledger_meta[str(p)],'status':'native_hardware_blocked','reason':row.get('reason')})
records=[]
for ledger,benchmark,model,state in targets:
 bp=BASE/benchmark/model/'plan.json';base=json.loads(bp.read_text());result=Path(base['reference_result']);config=result.parent/'config.json'
 rec={'benchmark':benchmark,'model':model,'selected_status':state['profiling_status'],'ledger':ledger_meta[str(ledger)],'retained_ledger_snapshot':meta(ledger_snapshots/ledger.name),'selected_profile_ids':state.get('profile_ids',[]),'selected_profile_provenance_sha256':state.get('profile_provenance_sha256'),'selected_configuration_id':state.get('profile_plan_configuration_id'),'selected_blocking_gates':state.get('profile_plan_blocking_gates'),'base_plan':meta(bp),'selected_result':meta(result),'expected_selected_result_sha256':base['reference_result_sha256'],'selected_config':meta(config),'expected_selected_config_sha256':base['raw_config_sha256'],'result_config_hashes_currently_match':sha(result)==base['reference_result_sha256']and sha(config)==base['raw_config_sha256'],'raw_plans':[]}
 for pp in sorted(RAW.glob('batch*/'+benchmark+'/'+model+'/plan.json')):
  d=json.loads(pp.read_text());provenance=pp.parent/'profile_provenance.json';prov=json.loads(provenance.read_text())if provenance.is_file()else None
  same=(d.get('configuration_id')==base['configuration_id']and d.get('reference_result_sha256')==base['reference_result_sha256']and d.get('raw_config_sha256')==base['raw_config_sha256'])
  p={'plan':meta(pp),'batch':pp.parts[-4],'configuration_id':d.get('configuration_id'),'reference_result_sha256':d.get('reference_result_sha256'),'raw_config_sha256':d.get('raw_config_sha256'),'same_selected_contract':same,'staged':d.get('staged',False),'blocking_gates':d.get('blocking_gates',[]),'provenance':meta(provenance)if provenance.is_file()else None,'raw_status':prov.get('status')if prov else 'no_attempt_provenance','execution_binding_count':len(d.get('execution_bindings',{})),'execution_bindings_inventory_sha256':hashlib.sha256(json.dumps(d.get('execution_bindings',{}),sort_keys=True,separators=(',',':')).encode()).hexdigest(),'bindings_are_prospective_not_rehashed_by_this_frontier_audit':True,'method_links':{k:v for k,v in d.items()if k in('application_replay_review','profile_window_review','isolated_assertion_review')},'live_finite_controller_matches':[]}
  root=str(pp.parents[2])
  for process in processes:
   if root not in process['plan_roots']or benchmark in process['excludes']or(process['includes']and benchmark not in process['includes']):continue
   args=process['argv'];required=[]
   if d.get('application_replay_review'):required.append('--application-replay-reviewed')
   if d.get('profile_window_review'):required.append('--profile-window-reviewed')
   hooks_match=all(x in args for x in required)
   # Parent followups may not forward these method flags; direct collector is
   # needed for method-specific scopes. Ordinary followups are explicit roots.
   p['live_finite_controller_matches'].append({'identity':process['controller_identity'],'script':process['script'],'explicit_plan_root':root,'required_method_flags':required,'required_method_flags_present':hooks_match,'pending_eligible':bool(same and d.get('staged')and not d.get('blocking_gates')and hooks_match and(prov is None or prov.get('status')=='started')),'basis':'Own live PID/starttime+exactargv root/filter/methodflag join; collector preserves any existing finalattempt. Does not establish future success.'})
  rec['raw_plans'].append(p)
 candidates=[(p,c)for p in rec['raw_plans']for c in p['live_finite_controller_matches']if c['pending_eligible']]
 if candidates:
  # Prefer latest reviewed fresh batch among explicit pending roots.
  p,c=max(candidates,key=lambda pc:pc[0]['batch']);rec['assignment']='pending_exact_live_finite_controller';rec['assigned_plan']=p['plan'];rec['assigned_batch']=p['batch'];rec['controller_identity']=c['identity'];rec['assignment_qualification']='Queued/started attempt only, no expected success or measuredcoverage imputed.'
 elif benchmark in('allreduce','ccl','p2p','pingpong','ring'):
  rec['assignment']='original_multidevice_contract_method_and_authorization_blocked';rec['method_review']=meta(Path('/tmp/roofline-completion-20261021/multidevice_roofline_method_review/method_matrix.json'));rec['assignment_qualification']='Original physicalGPU5+6 contract; no authorized GPU5 execution or generic communication kernelreplay method. allreduce private proposal does not equal execution assignment.'
  if benchmark=='allreduce':rec['private_preparation_only']=meta(Path('/tmp/roofline-completion-20261021/allreduce_two_gpu_proposal/DELIVERY_MANIFEST.json'))
 else:rec['assignment']='unassigned_requires_root_review';rec['assignment_qualification']='No exact live finite controller joined to ready samecontract rawplan in this snapshot; inspect before classifying source repair.'
 records.append(rec)
counts=collections.Counter(r['assignment']for r in records);unassigned=[{'benchmark':r['benchmark'],'model':r['model']}for r in records if r['assignment']=='unassigned_requires_root_review']
report={'schema_version':1,'observed_at':observed,'scope_statuses':sorted(STATUSES),'selected_scope_slots':len(records),'full_537_status_counts_context_only':dict(statuscounts),'assignment_counts':dict(counts),'unassigned_slots':unassigned,'records':records,'outside_scope_hardware_context':hardware,'process_snapshot':meta(HERE/'process_snapshot.json'),'conclusions':['Exactlive controller assignment is pending operational coverage only, not successful measurement or complete ROI.','Multidevice preparation remains permission/methodblocked, not hardwareunavailable or measured.','No automatic replay waiver, fixture/source changes or status/ledger changes made.'],'limitations':['Read-only asynchronous snapshot: process/ontology/rawstatus may advance immediately; PID/startticks protect identity interpretation only at recordedread.','Full executionbinding rehash/replay/runtime correctness is delegated to root collector before/after actualcapture; this is assignment census, not recertification.','No GPU/procdriver query, target import, build, test, plan staging, master/retry launch, foreign signal or sharedmetadata write.']}
dump(HERE/'review.json',report);print(json.dumps({'slots':len(records),'counts':dict(counts),'unassigned':unassigned,'assigned':[(r['benchmark'],r['model'],r.get('assigned_batch'),r.get('controller_identity'))for r in records if r['assignment']=='pending_exact_live_finite_controller'],'hardware_context_names':sorted(set(r['benchmark']for r in hardware))}))
