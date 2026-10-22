from pathlib import Path
import json,gzip,hashlib,collections,csv,fcntl,re,datetime
O=Path(__file__).resolve().parent; R=Path('/home/imo/HeCBench_julia'); X=json.loads((O/'snapshot_all_noncompleted.json').read_bytes()); sha=lambda b:hashlib.sha256(b).hexdigest()
rows=[]; sources={}; counters={}
source_names={
 'testSNAP':['study_common.jl','study_observer.cpp'],
 'simpleMultiDevice':['study_guard.jl','study_native.cu'],
 'rsc':['study_cuda.cu'], 'memcpy':['study_native.cu','study_common.jl'],
 'streamCreateCopyDestroy':['study_native.cu','study_common.jl','streamCreateCopyDestroy-julia-cuda.jl/main.jl','streamCreateCopyDestroy-julia-KA/main.jl','streamCreateCopyDestroy-julia-JACC/main.jl'],
 'snicit':['study_run_model.py','study_bridge.cu'], 'sparkler':['study_launch.py']}
for bench,names in source_names.items():
 for name in names:
  p=R/'src'/f'{bench}-julia'/name
  if p.exists():
   data=p.read_bytes();sources[str(p)]={'sha256':sha(data),'bytes':len(data),'lines':[{'line':i,'text':s}for i,s in enumerate(data.decode(errors='replace').splitlines(),1)if any(k in s for k in ['RAW_DIR','capture=','capture file open','visibility','CUDA_VISIBLE_DEVICES','pipeline(','authored_device_kernel_calls','snap_start','std::ofstream','__global__','<<<','tmpfile','Popen','subprocess'])]}
def read_attempt(a):
 p=a.get('files',{}).get('profile_provenance.json',{}).get('value',{});plan=a.get('files',{}).get('plan.json',{}).get('value',{})
 f=R/a['profile_id']/'counters.csv.gz';counter=None
 if f.exists():
  data=f.read_bytes(); h=hashlib.sha256();prefix=b'';size=0
  with gzip.open(f,'rb')as inp:
   for block in iter(lambda:inp.read(1<<20),b''):
    h.update(block);size+=len(block)
    if len(prefix)<5000:prefix=(prefix+block)[:5000]
  counter={'path':str(f),'compressed_sha256':sha(data),'raw_sha256':h.hexdigest(),'raw_bytes':size,'diagnostic_prefix':prefix.decode(errors='replace').split('"ID"')[0],'provenance_counter_sha_matches':p.get('counters_sha256')==h.hexdigest()}
  counters[str(f)]=counter
 log=a.get('logs',{}).get('execution.log');trials=[]
 if log:
  with gzip.open(log['path'],'rt',errors='replace')as inp:
   for line in inp:
    if line.startswith('STUDY_TRIAL '):
     head=line[:2200];t=re.search(r'"trial"\s*:\s*(\d+)',head);c=re.search(r'"correct"\s*:\s*(true|false)',head);tim=re.search(r'"time_seconds"\s*:\s*([0-9.eE+-]+)',head)
     trials.append({'trial':int(t[1])if t else None,'reported_correct':c[1]=='true'if c else None,'time_seconds':float(tim[1])if tim else None})
 return p,plan,counter,trials
for r in X['rows']:
 attempts=[]
 for a in r['attempts']:
  p,plan,counter,trials=read_attempt(a);v=p.get('profile',{});b=p.get('build',{})
  attempts.append({'profile_id':a['profile_id'],'selected':a['profile_id']in r['selected'],'latest':a['profile_id']==r['latest'],'manifest_sha256':a.get('manifest_sha256'),'plan_sha256':a.get('files',{}).get('plan.json',{}).get('raw_sha256'),'provenance_sha256':a.get('files',{}).get('profile_provenance.json',{}).get('raw_sha256'),'status':p.get('status'),'profile':{k:v.get(k)for k in ['returncode','stop_reason','wall_seconds','external_gpu_activity']},'build':{k:b.get(k)for k in ['returncode','stop_reason','wall_seconds']},'error':p.get('error'),'counter_parse_error':p.get('counter_parse_error'),'configuration_id':plan.get('configuration_id'),'raw_config_sha256':plan.get('raw_config_sha256'),'reference_result_sha256':plan.get('reference_result_sha256'),'counter':counter,'logs':{k:{a:b for a,b in d.items()if a not in ['head','tail']}for k,d in a.get('logs',{}).items()},'reported_trials':trials,'binding_source_checks':{path:{'expected_sha256':expected,'present_sha256':sources[path]['sha256'],'matches':sources[path]['sha256']==expected}for path,expected in plan.get('execution_bindings',{}).items()if path in sources}})
 latest=next((a for a in attempts if a['latest']),attempts[-1]if attempts else {})
 status=r['status'];n=r['benchmark'];stop=latest.get('profile',{}).get('stop_reason');err=latest.get('error')or''
 category='unknown_observed_failure';action='Retain failed evidence; diagnose exact source/process exit before any retry.';priority='review'
 if status=='native_hardware_blocked':category='source_reviewed_hardware_barrier';action='Preserve terminal native/device limitation; no repeat or substitute precision.';priority='hold'
 elif status=='not_started_plan_blocked':category='two_device_contract_permission_method_hold';action='GPU5 unauthorized; preserve full two-device contract, no single-device rewrite. Transport-only applicability is separate from authorization.';priority='hold'
 elif status=='roofline_not_applicable_no_device_kernels':category='reviewed_no_kernel_applicability';action='Retain reviewed runtime-only NA, not zero counters.';priority='terminal'
 elif status=='not_started_replay_method_blocked':category='concurrent_managed_memory_replay_hold';action='Do not kernel-replay concurrent host writes; separate concurrency-safe collection method needs source/scientific review.';priority='hold'
 elif n in ['memcpy','streamCreateCopyDestroy'] and status=='profile_completed_counter_derivation_failed':category='successful_runtime_only_capture_no_kernel_rows';action='CPU-only source/applicability adjudication candidate: runtime copy/lifecycle primary, rc0 and no-kernel CSV; verify full five checks/retained artifacts, do not synthesize zero roofline.';priority='P0_CPU_adjudication'
 elif n=='snicit':category='repaired_path_parent_runtime_failure_unresolved';action='Do not retry latest generic003 path. Preserve selected014 overlay; native245 / Julia255 empty diagnostic causes unknown. Review wrapper negative return propagation and child logs, then separate reviewed 2025 compatibility experiment only if warranted.';priority='P2_source_diagnosis'
 elif n=='sparkler':
  category='mpi_shared_memory_guard_or_binding_failure';action='Native024 MPI open(2) permission error then timeout: prove fresh authorized MPI shared-memory directory/env without weakening Landlock. Julia024 compiler_manifest binding changed before profile: reconstruct fresh role-reviewed compiler manifest, never drop binding. Retain rank4 checks.';priority='P1_source_path_MPI'
 elif stop=='external_gpu_activity_detected_own_profile_stopped':category='observed_foreign_gpu_stop';action='One root-reviewed fresh exact-latest-method retry after idle, skip already qualified. Preserve custom source copies/checkers/flags; no new master or repeated039/041. Foreign identity ownership remains as observed, not inferred false positive.';priority='P0_finite_recovery'
 elif n=='testSNAP':category='capture_root_guard_rejection';action='Copy Julia study_common.jl path-only: retain canonical read-only ROOT and includes; redirect capture to fresh private root. Preserve observer, full3 cases/5 checks; new exact forward/inverse source proof and plan bindings required.';priority='P1_path_copy'
 elif n=='simpleMultiDevice':category='authorized_uuid_rejected_by_literal_device_guard';action='All4 selected argv1 is one-device scope: exact guard-only copied-source allow GPU6 UUID alongside existing6; preserve device-count/PCI/UUID validation and do not expose/query GPU5. Full source inverse proof required.';priority='P1_visibility_guard'
 elif n=='rsc':category='historical_compiletime_capture_path_exit3';action='Native compile macro RAW_DIR is historical output. Fresh build flag-only RAW_DIR under scratch, fixture/oracle/DSO unchanged. Exit3 also other source branches: exact failure site not observed, recovery unproven.';priority='P1_build_output_role'
 elif n=='secp256k1':category='native_cpu_build_deadline';action='Build stopped at180.59s with nvcc Terminated, no compiler rejection proved. Fresh exact build flags/source with separately bounded larger CPU build deadline; do not infer JIT/framework failure.';priority='P1_build_budget'
 elif stop=='wall_timeout':
  category='observer_operational_deadline';action='No automatic retry. Source-bound bounded deadline experiment up to3600, preserving all5 checks and method. Assess source progress and payload reserve before root approval.';priority='P1_deadline_review'
  if n=='slu':action='Use latest018 bounded per-function selector, not old007 geometry census; genuine full ASIC independent case0 check passed during warmup. Increase explicit300s deadline to root-reviewed<=3600, preserve4GB replay context/payload budget and full5 source checks.'
  elif n=='tgvnn':action='Latest600s retained two complete scientific trials (~140–161s each); fresh exact039 source/method with explicit3600s budget is plausible. Preserve all source steps, warmup and full5 checked outputs; budget ~1.44GB/model primary plus aux.'
  elif n=='spsort':action='All4 twice timed out600s; workspace/vendor stages observed but no complete trial. Review exact cuSPARSE specialization/geometry census and replay progress; bounded per-function alternative requires source/ROI method review before retry, not blind third same600s.'
  elif n=='dispatch':action='Source execution-only~1.1M launches/trial; observer600s timeout with checked progress. Review max3600 runtime and empty-kernel applicability/selector before one exact-source retry; never substitute shorter workload or fabricated FP point.'
 rows.append({'benchmark':n,'model':r['model'],'selected_status':status,'selected_profile_ids':r['selected'],'latest_profile_id':r['latest'],'category':category,'priority':priority,'proposed_next_action':action,'ledger':r['ledger'],'planning_gates':r['state'].get('profile_plan_blocking_gates',[]),'attempts':attempts})
with Path('/tmp/hecbench-roofline-metadata.lock').open('r')as lock:
 fcntl.flock(lock,fcntl.LOCK_SH)
 changed=[r['ledger']['path']for r in rows if sha(Path(r['ledger']['path']).read_bytes())!=r['ledger']['sha256']]
 prog=R/'ontology/studies/v100_roofline_537/PROGRESS.json';progress_after=sha(prog.read_bytes())
summary={'schema_version':1,'created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'saved current-selected and latest attempts; CPU-only source/log audit, no runtime recovery performed','snapshot_sha256':sha((O/'snapshot_all_noncompleted.json').read_bytes()),'progress_updated_at':X['progress'].get('updated_at'),'progress_sha256_before':X['progress_sha256'],'progress_sha256_after':progress_after,'ledger_drift':changed,'selected_status_counts':dict(collections.Counter(r['selected_status']for r in rows)),'latest_taxonomy_counts':dict(collections.Counter(r['category']for r in rows)),'priority_counts':dict(collections.Counter(r['priority']for r in rows)),'total_noncompleted_slots':len(rows),'future_execution_authorized_by_this_report':False,'no_new_master_or_repeated_finite_rounds':True,'no_gpu5_permission':True,'no_causal_or_whole_ROI_claim':True}
(O/'triage_all114.json').write_text(json.dumps({'summary':summary,'slots':rows},indent=2)+'\n')
(O/'source_role_evidence.json').write_text(json.dumps(sources,indent=2)+'\n')
with (O/'triage_all114.csv').open('w',newline='')as f:
 w=csv.DictWriter(f,fieldnames=['benchmark','model','selected_status','category','priority','latest_profile_id','proposed_next_action']);w.writeheader();w.writerows({k:r[k]for k in w.fieldnames}for r in rows)
(O/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
