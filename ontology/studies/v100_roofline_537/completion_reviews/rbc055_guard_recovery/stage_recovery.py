from pathlib import Path
import json,hashlib,copy,datetime,shutil
P=Path;O=P(__file__).resolve().parent;W=P('/home/imo/HeCBench_julia');OLD=P('/tmp/hecbench-roofline-v100-20261020/batch050-rbc-retained-samples');NEW=P('/tmp/hecbench-roofline-v100-20261020/batch055-rbc-retained-guard-retry');B='rayleighBenardConvection';OLDREVIEW=P('/tmp/roofline-completion-20261021/rbc_batch050_deadline_review/profile_window_review.json');NEWREVIEW=O/'profile_window_review.json';COLLECTOR='32a9cfeba983dc3ae8e5d9b8bb85ca4e7736821be0a9c206bb57b80c7471da4c'
def sha(p):return hashlib.sha256(P(p).read_bytes()).hexdigest()
def bind(p):p=P(p);return {'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)}
def dump(p,d):p.write_text(json.dumps(d,indent=2)+'\n')
def move(v):
 if isinstance(v,str):return v.replace(str(OLD),str(NEW)).replace(str(OLDREVIEW),str(NEWREVIEW))
 if isinstance(v,list):return [move(x)for x in v]
 if isinstance(v,dict):return {move(k):move(x)for k,x in v.items()}
 return v
if NEW.exists()or NEWREVIEW.exists():raise ValueError('Fresh paths required')
if sha(W/'tools/v100_roofline_collect.py')!=COLLECTOR:raise ValueError('Parent-pinned current collector drift')
d=json.loads(OLDREVIEW.read_text());n=move(d);rows=[];plans=[];newfiles=[];verified={}
for m in ['cuda','cuda_jl','ka','jacc']:
 oldslot=OLD/B/m;slot=NEW/B/m;scratch=slot/'scratch';z=json.loads((oldslot/'plan.json').read_text());rec=d['models'][B+'/'+m]
 if z['blocking_gates']or z['runtime_asset_bytes']or z['runtime_assets']or not z['staged']:raise ValueError('Unexpected original RBC plan gates/assets')
 if z['profile_window_review']!={'path':str(OLDREVIEW),'sha256':sha(OLDREVIEW),'entry':B+'/'+m}:raise ValueError('Old review link mismatch')
 if rec['selected_window']!={'retained_trial':2,'zero_based_step':240,'ordinal_step':241,'whole_scientific_timestep':True,'Nu_conditional_required':True,'Nu_samples_before':1,'Nu_samples_after':2,'completed_unprofiled_steps_in_same_trial_before':240,'total_controller_runs':6,'steps_each_run':1000,'total_steps':6000,'checked_trials':5,'warmup_runs':1}:raise ValueError('Exact full controller/window required')
 if rec['source_expected_authored_dispatch_counts']!=[3,6,2,11,5,0,1]or rec['source_expected_authored_dispatches']!=28:raise ValueError('Authored source ledger mismatch')
 if rec['capture_all_launches_in_window']is not False or rec['filter_mode']!='per-launch-config'or rec['kernel_name_base']!='mangled'or rec['kernel_id']!='::regex:.*:1':raise ValueError('Original partial sample method required')
 if rec['original_GPU_function_bytes_sha256']!=rec['copied_GPU_function_bytes_sha256']or len(rec['original_GPU_function_bytes_sha256'])!=7:raise ValueError('Exact kernel math bytes required')
 if rec['runtime_validated']is not False:raise ValueError('Preserve prospective method qualification')
 if rec['execution_bindings']!=z['execution_bindings']:raise ValueError('Review/plan execution binding equality required')
 for path,h in z['execution_bindings'].items():
  if path not in verified:verified[path]=sha(path)
  if verified[path]!=h:raise ValueError('Original input drift '+path)
 for f in ['scratch/profile_outputs','scratch/ncu_tmp','scratch/profile_source']:(slot/f).mkdir(parents=True,exist_ok=False)
 mappings=[]
 for path,h in z['execution_bindings'].items():
  if path.startswith(str(OLD)+'/'):
   target=P(move(path));target.relative_to(scratch);bs=P(path).read_bytes()
   if str(OLD).encode()in bs:raise ValueError('Cannot preserve source bytes under path rebase '+path)
   target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(bs)
   if sha(target)!=h:raise ValueError('Source copy byte drift')
   mappings.append({'old_path':path,'new_path':str(target),'unchanged_sha256':h});newfiles.append(bind(target))
 q=move(z);q['profile_window_review']['sha256']=None
 # Preserve exact050 observer3600 proposal metadata; no deadline/policy change.
 if q.get('observer_deadline_proposal',{}).get('seconds')!=3600:raise ValueError('Exact050deadline3600required')
 if q['execution_bindings']!=n['models'][B+'/'+m]['execution_bindings']:raise ValueError('Dictionary binding keys not rebased consistently')
 if q['build_command'][1]!=str(W/'tools/v100_profile_write_guard.py'):raise ValueError('Absolute original writeguard required')
 for fname,key in [('configured_source.cuh','copied_header_sha256'),('full_study.cu','copied_fullstudy_sha256')]:
  if sha(scratch/'profile_source'/fname)!=rec[key]:raise ValueError('Reviewed marker/controller source changed')
 result=P(z['reference_result']);config=result.with_name('config.json')
 if sha(result)!=z['reference_result_sha256']or sha(config)!=z['raw_config_sha256']:raise ValueError('Selected contract drift')
 plans.append((slot,q));rows.append({'model':m,'original_plan':bind(oldslot/'plan.json'),'new_plan_path':str(slot/'plan.json'),'configuration_id':z['configuration_id'],'reference_result_sha256':z['reference_result_sha256'],'raw_config_sha256':z['raw_config_sha256'],'private_source_path_mappings':mappings,'canonical_execution_bindings_unchanged':{p:h for p,h in z['execution_bindings'].items()if not p.startswith(str(OLD)+'/')},'copied_header_sha256':rec['copied_header_sha256'],'copied_fullstudy_sha256':rec['copied_fullstudy_sha256'],'kernel_function_bytehashes':rec['original_GPU_function_bytes_sha256'],'runtime_asset_bytes':0,'partial_sample_method_unchanged':True,'runtime_validated':False})
dump(NEWREVIEW,n);newreviewsha=sha(NEWREVIEW)
for slot,q in plans:q['profile_window_review']['sha256']=newreviewsha;dump(slot/'plan.json',q)
for row in rows:row['new_plan']=bind(row.pop('new_plan_path'))
dump(NEW/'summary.json',[{'model':row['model'],'plan':row['new_plan']['path'],'sha256':row['new_plan']['sha256'],'review_sha256':newreviewsha}for row in rows])
# Snapshot only the current small log; do not mutate/read trial payload artifacts.
log=OLD/B/'cuda/execution.log';snapshot=O/'batch050_cuda_execution_failed.log';snapshot.write_bytes(log.read_bytes());lines=snapshot.read_text().splitlines();trialrows=[json.loads(l.removeprefix('STUDY_TRIAL '))for l in lines if l.startswith('STUDY_TRIAL ')];markers=[l for l in lines if l.startswith('RBC_PROFILE_WINDOW_')]
cmd=['python3',str(W/'tools/v100_roofline_collect.py'),'--source-root',str(W),'--plan-root',str(NEW),'--profile-window-reviewed','--skip-qualified-selected','--timeout','3600','--retain-auxiliary']
report={'schema_version':1,'created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'purpose':'One finite RBC operational guard retry after050native interrupted. Source/method/deadline3600 unchanged; six full1000step runs and all5checks retained, not shortened.','prior_review':bind(OLDREVIEW),'new_review':bind(NEWREVIEW),'helper':bind(P(__file__)),'prior_recovery_algorithm':bind('/tmp/roofline-completion-20261021/rbc_batch050_deadline_review/stage_recovery.py'),'prior_instrumentation_algorithm':bind('/tmp/roofline-completion-20261021/rbc_batch044_recovery/prior_stage_RBC_samples.py'),'current_collector':bind(W/'tools/v100_roofline_collect.py'),'current_collector_sha256_expected':COLLECTOR,'path_mapping':{'prefixes':{str(OLD):str(NEW),str(OLDREVIEW):str(NEWREVIEW)},'method':'Recursive string replacement in BOTH dictionary keys and values, lists, commands, compiler_argv, unified-diff path labels and review links. Private source files copied byte-for-byte; canonical/helper/history/source/config/result paths remain unchanged. Review SHA recomputed only after all new model records ready; plan links then finalized.','review_semantics_equality_after_path_normalization':n==move(d)},'new_plans':rows,'source_review':{'total_steps':6000,'warmup_runs':1,'full_checked_trials':5,'retained_trial':2,'zero_based_step':240,'Nu_before':1,'Nu_after':2,'authored_dispatch_counts':[3,6,2,11,5,0,1],'authored_dispatches':28,'marker_entry_exit_asserted_once':True,'all7_GPU_function_bytes_unchanged':True,'header_controller_source_bytes_unchanged':True,'selection':'First per mangled specialization/launch geometry within marked window; partial samples, not complete timestep/ROI.','actual_NCU_launch_count':None,'full_primary_oracles_and_writeguards_preserved':True},'observer_budget':{'prior_seconds':3600,'proposed_seconds':3600,'scope':'Collector observer deadline only; source runtime/control/input/math/checks unchanged.','evidence':'050native actualownershipguardinterruption after2fullchecks, notobserverdeadline ormathfailure. Same3600deadline retained; no completion forecast.'},'prior050_failed_partial_observation':{'original_path':str(log),'snapshot':bind(snapshot),'trial_rows':trialrows,'marker_lines':markers,'qualification':'Read-only point-in-time log snapshot; active050 plans/controllers untouched, future outcomes must be preserved by parent before any retry. Partial profiled elapsed values do not replace primary timings.'},'root_only_future_invocation':cmd,'launch_conditions':['Root must wait exactowned050controller1326488:start77008700, preservealloutcomes, independentlyreview055andadoptfrontier beforeonefinitecollector. No worker cancellation orforeignsignals.','skip-qualified-selected preserves a completed selected050 capture; stage creation alone never requires redundant GPU execution.','Use exact absolute source root/collector executable, existing GPU6 UUID/foreignprocess refusal/global lock/Landlock and source recheck.','No new calibration/collector changes, no full-step aggregate, no short checks or modified mathematical iteration budget.','Cold replay/caches/instrumentation context must be labeled separately from primary five-trial timings; actual first-per-window filter behavior still requires observed capture.'],'GPU_runs':0,'build_runs':0,'tests':0,'runtime_assets_copied':0,'shared_file_changes':False,'shared_repair_not_translation':True}
priorprov=OLD/B/'cuda/profile_provenance.json';(O/'batch050_cuda_failed_provenance.json').write_bytes(priorprov.read_bytes());pv=json.loads(priorprov.read_text())
proc=P('/proc/1326488');fields=(proc/'stat').read_text().rsplit(')',1)[1].split();argv=(proc/'cmdline').read_bytes().split(b'\0');argv=[x.decode()for x in argv if x]
if fields[19]!='77008700'or proc.stat().st_uid!=__import__('os').getuid():raise ValueError('Exactowned050controlleridentitychanged')
report['predecessor_controller']={'pid':1326488,'start_ticks':'77008700','uid':proc.stat().st_uid,'argv':argv,'identity':'1326488:77008700','snapshot_time':datetime.datetime.now(datetime.timezone.utc).isoformat()}
report['actual_prior_failure']={'provenance_original':bind(priorprov),'provenance_snapshot':bind(O/'batch050_cuda_failed_provenance.json'),'status':pv['status'],'profile':pv.get('profile'),'observed_successful_trial_ids':[x['trial']for x in trialrows],'observed_profiled_trial_seconds':[x.get('time_seconds')for x in trialrows],'qualifications':['Theseareprofilingdiagnosticelapsedvalues,notprimarytimings.','Historicaltrigger1397413ownershipunknownafterexit; no retrospectiveUIDattribution.','ParentreportedcurrentforeignGPU6RSBenchPID1408488UID11887at21:20:44; thisworkerperformednoGPUquery/noindependentobservation.','Prior050failedreceipt andactiveJuliaattempts retained; noforeignsignal, mathfailureclaim orwaiver.']}
report['retry_policy']={'kind':'one_explicit_finite_custom_operational_retry','method_or_timeout_change':False,'max_new_collector_invocations':1,'wait_for_exact_controller':'1326488:77008700','skip_qualified_selected':True,'generic039_041_custom_method_hold_preserved':True,'root_review_and_frontier_adoption_required':True}
dump(O/'recovery_review.json',report);dump(O/'root_invocation.json',cmd)
print(json.dumps({'staged':4,'review':bind(O/'recovery_review.json'),'profile_window_review':bind(NEWREVIEW),'plans':[row['new_plan']for row in rows],'observed_partial_trial_ids':[x['trial']for x in trialrows],'GPU_runs':0,'build_runs':0,'tests':0},indent=2))
