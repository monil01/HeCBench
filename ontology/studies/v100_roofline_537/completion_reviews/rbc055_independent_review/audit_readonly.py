from pathlib import Path
import hashlib,json,re,datetime
O=Path(__file__).resolve().parent;P=Path('/tmp/roofline-completion-20261021/rbc055_guard_recovery');R=Path('/home/imo/HeCBench_julia');OLD=Path('/tmp/hecbench-roofline-v100-20261020/batch050-rbc-retained-samples');NEW=Path('/tmp/hecbench-roofline-v100-20261020/batch055-rbc-retained-guard-retry');B='rayleighBenardConvection'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):return {'path':str(p),'sha256':sha(p)}
def j(p):return json.loads(Path(p).read_text())
def functions(p):
 text=Path(p).read_text();out={}
 for m in re.finditer(r'__global__\s+void\s+(\w+)\s*\([^;]*?\)\s*\{',text,re.S):
  n=m.end();depth=1
  while depth:
   c=text[n];depth+=(c=='{')-(c=='}');n+=1
  out[m[1]]=hashlib.sha256(text[m.start():n].encode()).hexdigest()
 return out
def remap(v):
 if isinstance(v,str):return v.replace(str(OLD),str(NEW)).replace('/tmp/roofline-completion-20261021/rbc_batch050_deadline_review/profile_window_review.json',str(P/'profile_window_review.json'))
 if isinstance(v,list):return [remap(x)for x in v]
 if isinstance(v,dict):return {remap(k):remap(x)for k,x in v.items()}
 return v
checks=[];blockers=[]
def check(name,ok,details=None):
 checks.append({'name':name,'passed':bool(ok),'details':details})
 if not ok:blockers.append(name)
manifest=j(P/'DELIVERY_MANIFEST.json')
for e in manifest['private_files']:check('private_inventory:'+e['path'],sha(P/e['path'])==e['sha256'])
for e in manifest['fresh_raw_plan_files']:check('raw_inventory:'+e['path'],sha(e['path'])==e['sha256'])
review=j(P/'launch_review.json');window=j(P/'profile_window_review.json');pins=j(P/'frozen_method_review.json')
for path,s in review['readonly_bindings'].items():check('binding:'+path,sha(path)==s)
collector=P/'frozen_collect_32a9.py';check('frozen_collector_verbatim_actual050',collector.read_bytes()==(OLD/B/'cuda/collector_source.py').read_bytes() and sha(collector)=='32a9cfeba983dc3ae8e5d9b8bb85ca4e7736821be0a9c206bb57b80c7471da4c')
models=[]
for m in ['cuda','cuda_jl','ka','jacc']:
 np=NEW/B/m/'plan.json';op=OLD/B/m/'plan.json';n=j(np);old=j(op);mapped=remap(old)
 # Exactly added read-only method pins plus recomputed marker-review hash are permitted.
 added={k:v for k,v in n['execution_bindings'].items()if k not in mapped['execution_bindings']}
 permitted={**pins['execution_bindings'],str(P/'frozen_method_review.json'):sha(P/'frozen_method_review.json')}
 check(m+':only_frozen_method_pins_added',all(permitted.get(k)==v for k,v in added.items()))
 mapped['execution_bindings'].update(added);mapped['profile_window_review']['sha256']=sha(P/'profile_window_review.json')
 check(m+':full_plan_equality_after_path_rebase_pins',mapped==n)
 stage=window['models'][B+'/'+m]
 check(m+':review_execution_bindings',stage['execution_bindings']==n['execution_bindings'])
 for name in ['configured_source.cuh','full_study.cu']:
  a=OLD/B/m/'scratch/profile_source'/name;b=NEW/B/m/'scratch/profile_source'/name
  check(m+':exact050source:'+name,a.read_bytes()==b.read_bytes())
 header=NEW/B/m/'scratch/profile_source/configured_source.cuh';original=R/'src'/f'{B}-julia/configured_source.cuh'
 f=functions(header);check(m+':seven_GPU_bodies_original',len(f)==7 and f==functions(original) and f==stage['original_GPU_function_bytes_sha256'])
 text=header.read_text()
 for c in reversed(stage['source_changes']):text=text.replace(c['new'],c['old'],1)
 check(m+':inverse_instrumentation_recovers_source',text==original.read_text())
 check(m+':full_six_runs_and_five_checks',stage['selected_window']=={'retained_trial':2,'zero_based_step':240,'ordinal_step':241,'whole_scientific_timestep':True,'Nu_conditional_required':True,'Nu_samples_before':1,'Nu_samples_after':2,'completed_unprofiled_steps_in_same_trial_before':240,'total_controller_runs':6,'steps_each_run':1000,'total_steps':6000,'checked_trials':5,'warmup_runs':1})
 check(m+':dispatch_count_28',sum(stage['source_expected_authored_dispatch_counts'])==stage['source_expected_authored_dispatches']==28 and stage['source_expected_authored_dispatch_counts']==[3,6,2,11,5,0,1])
 check(m+':selection_is_partial',stage['kernel_id']=='::regex:.*:1'and stage['filter_mode']=='per-launch-config'and stage['kernel_name_base']=='mangled'and stage['capture_all_launches_in_window']is False and stage['profile_from_start']is False)
 check(m+':GPU6_and_no_fixture_copies',n['environment_overrides']['CUDA_VISIBLE_DEVICES']=='6'and n['runtime_asset_bytes']==0 and n['runtime_assets']==[])
 models.append({'model':m,'plan':bind(np),'old_plan':bind(op),'configured_header':bind(header),'full_controller':bind(header.with_name('full_study.cu')),'source_config_sha256':n['raw_config_sha256'],'source_result_sha256':n['reference_result_sha256'],'added_readonly_method_bindings':added,'build_command':n['build_command'],'target_command':n['profile_target_command']})
failed=j(P/'batch050_cuda_failed_provenance.json');check('050_stopped_operational_not_math',failed['status']=='profile_process_failed'and failed['profile']['stop_reason']=='external_gpu_activity_detected_own_profile_stopped',{'status':failed['status'],'stop_reason':failed['profile']['stop_reason']})
check('follower_own_hash',sha(P/'wait_then_collect.py')==review['follower_sha256'])
check('absolute_source_root_and_frozen_collector',review['collector_argv']==j(P/'root_invocation.json') and review['collector_argv'][1]==str(collector)and review['source_root']==str(R))
check('old_version_saved_identity',pins['observed050_NCU']['version']=='NVIDIA (R) Nsight Compute Command Line Profiler\nCopyright (c) 2018-2023 NVIDIA Corporation\nVersion 2022.4.1.0 (build 32308335) (public-release)\n')
report={'schema_version':1,'reviewed_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'root_launch_admissible':not blockers,'approval_scope':'Static source/binding/controller review permits root one finite future operational retry only; no observed runtime pass or authorization for another GPU. Root owns actual adoption/launch and supervision.','blockers':blockers,'checks':checks,'models':models,'packet':bind(P/'DELIVERY_MANIFEST.json'),'launch_review':bind(P/'launch_review.json'),'follower':bind(P/'wait_then_collect.py'),'frozen_collector':bind(collector),'observations':[
'Frozen collector standalone source imports stdlib only; dynamic classifier/calibration/guard resolve via explicit absolute repository root, __file__ only own source snapshot. No dependence on package-relative tools imports.',
'Follower verifies exact launch-review SHA, own SHA and71 read-only paths before waiting and again before collector. proc.stat index19 is Linux starttime after comm; PID reuse/absent/zombie ends waiting without signals. Same-generation UID/argv mismatch fails closed. Follower never queries GPU or signals any process.',
'Predecessor exit is not proof GPU idle; unchanged collector independently obtains existing GPU lock, GPU6 UUID/PCI checks and idle/foreign-process checks. Child session-only termination on guard violation; no foreign signals. No use of GPU5.',
'Follower wait cap86400 applies only to predecessor. Collector run timeout3600 applies separately per build/profile and excludes idle wait. wait_idle_snapshot remains unbounded as in050; root finite supervisor must own operational deadline. No completion forecast or idle exclusivity assertion.',
'Fresh child writes restricted to model attempt and existing disposable Julia caches; build also nested scratch-only guard. Source fixture directory is an existing read-only input: controller create_directories on that existing path needs no creation; prepare arg0 excludes fixture writes. Native outputs are relative scratch/output_trialN.bin; Julia calls CABI prepare0 and retains full outputs. NCU TMP/cache directories are private scratch.',
'Observer uses cold-cache all and clocknone old2022; profiler wrapper and targetELF byte hashes pinned, follower requires PATH resolution matches old wrapper immediately before launch; no2025 receipt/calibration inferred. Runtime environment inherits same old Julia LD injection; no environment change proposed.',
'Six whole1000-step runs are preserved: full warmup trial0, five checks1..5, exact input bytes, full field/dt/Nu independent budgets. Retained trial2 zero-basedstep240 encloses RK3 and second Nu event; partial first-mangled-specialization/geometry selector, not all launches/fullROI. Finite budget failure remains failure.',
'050 snapshot contains passing full checks1/2 and marker pair before operational guard stop; no numerical failure diagnosed or waived. All old outcomes/raw counter artifacts remain preserved. skip-qualified-selected can avoid completed050 model recapture; partial/failure never promoted by mere returncode.',
'Unprivileged future follower writes only its private receipt/log and executes unchanged collector; no shared source/metadata edits by this review. Collector records own perattempt provenance as original design. Root adoption/frontier controller is outside this packet review.'
],'qualifications':['Static audit only; no imports, builds, tests, GPU queries or launches.','Actual runtime fullfivechecks/marker/counter admissibility must be evaluated after any future run.','Root should recheck all bindings and exact process identity immediately at adoption; TOCTOU cannot be eliminated by a prior read-only review.','Fresh output-root existence is intentional staged source/plans only; no profile provenance or prior trial evidence should be overwritten.'],'GPU_queries':0,'GPU_runs':0,'build_runs':0,'tests':0,'shared_edits':False}
(O/'independent_review.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'root_launch_admissible':report['root_launch_admissible'],'checks':len(checks),'blockers':blockers,'review_sha256':sha(O/'independent_review.json')}))
