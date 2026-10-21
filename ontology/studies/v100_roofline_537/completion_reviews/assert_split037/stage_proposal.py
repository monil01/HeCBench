"""CPU-only fresh proposal staging. Does not import or execute phase helper/targets."""
import copy,difflib,hashlib,json,pathlib,shutil
P=pathlib.Path
ROOT=P('/home/imo/HeCBench_julia');OWN=P('/tmp/roofline-completion-20261021/assert_split037');RAW=P('/tmp/hecbench-roofline-v100-20261020/batch037-assert-isolated-phases');HELP=OWN/'assert_split_phases.py';REVIEW=OWN/'isolated_assertion_review.json'
def sha(p):return hashlib.sha256(P(p).read_bytes()).hexdigest()
def dump(p,d):p.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
if RAW.exists():raise RuntimeError('fresh batch required')
rows={};pending=[]
for model in ['cuda_jl','ka','jacc']:
 base=OWN.parent/'remaining_infra/current_base_v7/assert'/model/'plan.json';old=json.loads(base.read_text());d=copy.deepcopy(old)
 slot=RAW/'assert'/model;scratch=slot/'scratch';scratch.mkdir(parents=True);(scratch/'profile_outputs').mkdir();(scratch/'ncu_tmp').mkdir()
 oldscratch=old['scratch_cwd'];d['scratch_cwd']=str(scratch)
 d['environment_overrides']={k:v.replace(oldscratch,str(scratch)) if isinstance(v,str) else v for k,v in old['environment_overrides'].items()}
 args=old['profile_target_command'];julia_index=next(i for i,x in enumerate(args) if x in ['julia','/home/imo/.juliaup/bin/julia'] or x.endswith('/bin/julia'));positive=args[julia_index:]
 positive[0]='/home/imo/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia'
 if any('--negative' in x for x in positive):raise RuntimeError('positive command not original default')
 d['profile_target_command']=positive;d['build_command']=[];entry='assert/'+model
 bindings={k:v for k,v in old['execution_bindings'].items()}
 for p,h in bindings.items():
  if sha(p)!=h:raise RuntimeError('base binding drift '+p)
 # Explicit environment/project manifests plus exact ELF, source wrapper/entry.
 for p in [HELP,OWN/'stage_proposal.py',base,P(positive[0]),ROOT/'src/assert-julia/run_study.py',ROOT/'tools/v100_profile_write_guard.py']:
  bindings[str(p)]=sha(p)
 for arg in positive:
  if arg.startswith('--project='):
   project=P(arg.split('=',1)[1])
   for name in ['Project.toml','Manifest.toml','LocalPreferences.toml']:
    if (project/name).is_file():bindings[str(project/name)]=sha(project/name)
 planpath=slot/'plan.json';specpath=slot/'phase_spec.json';marker='reviewed_ASSERT_split_phases_037_'+model
 spec={'schema_version':1,'benchmark':'assert','model':model,'positive_command':positive,'scratch_cwd':str(scratch),'authorization_marker':marker,'execution_bindings':bindings,'original_wrapper':str(ROOT/'src/assert-julia/run_study.py'),'pre_receipt':str(slot/'negative_prechecks.json'),'post_receipt':str(slot/'joined_execution_contract.json'),'plan_path':str(planpath),'original_configuration_id':d['configuration_id'],'reference_result_sha256':d['reference_result_sha256'],'raw_config_sha256':d['raw_config_sha256']}
 dump(specpath,spec);bindings=copy.deepcopy(bindings);bindings[str(specpath)]=sha(specpath)
 pre=['python3',str(HELP),'pre','--spec',str(specpath)];post=['python3',str(HELP),'post','--spec',str(specpath),'--positive-outcome',str(slot/'positive_profile_outcome.json')]
 row={'benchmark':'assert','model':model,'source_reviewed':True,'runtime_validated':False,'configuration_id':d['configuration_id'],'reference_result_sha256':d['reference_result_sha256'],'raw_config_sha256':d['raw_config_sha256'],'execution_bindings':bindings,'positive_command':positive,'pre_profile_check_command':pre,'post_profile_check_command':post,'authorization_environment':{'HECBENCH_ASSERT_PHASE_AUTHORIZATION':marker},'expected_negative_context_count':5,'expected_negative_status':710,'expected_positive_PERF_TRIAL_count':5,'negative_receipt':spec['pre_receipt'],'joined_receipt':spec['post_receipt'],'positive_outcome_path':str(slot/'positive_profile_outcome.json'),'phase_spec':{'path':str(specpath),'sha256':sha(specpath)},'source_semantics':'Original negative entry first checks valid N64 then catches expected device assertion N60 with status710/_exit0. Original positive entry performs two warmups and five synchronized two-kernel PERF_TRIAL records. No kernels/source arithmetic/geometry or primary numerical/timing criteria changed. Placement of checks relative to profiler changes only profiling orchestration.','cause_of_prior_child255':'unknown; isolation is a reviewed unproven recovery experiment, not a diagnosis','scope':'execution-only; no numerical output or equality claim. Two positive kernel types sampled; warmup may be selected and no whole-ROI coverage claim.','required_parent_evidence':['GPU6 UUID/PCI verification','exclusive common GPU lock across all phases','child Landlock enforced on each phase and NCU','before/after source hashes around pre/positive/post','all five precheck process returncodes and full raw bytes','profile session quiescence and actual rc/stop reason','mandatory successful CPU postvalidation before acceptance'],'fixed_retained_files':['negative_prechecks.json','positive_profile_outcome.json','joined_execution_contract.json'],'retained_globs':['negative_raw/*.stdout','negative_raw/*.stderr','negative_raw/*.combined.log'],'counts':{'runtime_negative_processes':5,'positive_profile_processes':1,'compiler_build_processes':0,'device_JIT_invocations':'unknown_do_not_infer'}}
 rows[entry]=row;d['execution_bindings']=bindings;d['pre_profile_check_command']=pre;d['post_profile_check_command']=post;d['blocking_gates']=['requires_isolated_assertion_collector_phase_hooks'];d['staging_status']='source_reviewed_waiting_explicit_isolated_assertion_collector_optin';d['isolated_assertion_review']={'path':str(REVIEW),'sha256':None,'entry':entry};d['review']={'path':str(REVIEW),'sha256':None,'entry':{'source_reviewed':True,'runtime_validated':False,'reason':row['source_semantics'],'additional_current_readonly_input_bindings':bindings}};d['commands_executed']=False
 d['isolated_assertion_proposal']={'original_plan':str(base),'original_plan_sha256':sha(base),'original_profile_command':old['profile_target_command'],'no_GPU_build_or_tests':True,'original_accepted_contract_unchanged':True,'pending_integration':'parent-owned collector opt-in must validate source/hash scopes, run pre and post under lock/guard, archive receipts, forbid profile if precheck fails; post failure invalidates profile attempt'}
 pending.append((d,old,slot,base))
review={'schema_version':1,'review_kind':'assert_Julia_three_model_isolated_negative_prechecks_original_positive_profile_mandatory_CPU_join','source_reviewed':True,'runtime_validated':False,'models':rows,'positive_outcome_schema':{'plan_sha256':'actual plan SHA','command':'direct exact positive argv from reviewed plan','returncode':'actual NCU process returncode; must be0','stop_reason':'actual stop reason; must be null','own_session_quiescent':True,'guarded_under_GPU_lock':True,'execution_log':{'path':'slot/execution.log','sha256':'actual full bytes SHA','bytes':'actual byte count'}},'retention':'Retain all raw negative streams/receipts even on rejection or timeout, plus per-phase launcher log, signed status, command, source/guard checks. Slot-top writes stay after scratch cleanup. All precheck processes are runtime verification work, not compiler/build/JIT counts.','uncertainty':'Historical child exit255 empty stdout cause remains unproved. This proposal has not been GPU executed.'}
dump(REVIEW,review)
summary=[]
for d,old,slot,base in pending:
 for key in ['review','isolated_assertion_review']:d[key]['sha256']=sha(REVIEW)
 d['execution_bindings'][str(REVIEW)]=sha(REVIEW)
 dump(slot/'plan.json',d)
 dump(slot/'plan_diff.json',{'base_plan':str(base),'base_sha256':sha(base),'new_plan_sha256':sha(slot/'plan.json'),'diff':''.join(difflib.unified_diff(json.dumps(old,indent=2).splitlines(True),json.dumps(d,indent=2).splitlines(True),fromfile=str(base),tofile=str(slot/'plan.json'))),'source_files_modified':[],'GPU_runs':0,'build_runs':0,'tests_run':0})
 summary.append({'benchmark':'assert','model':d['model'],'plan':str(slot/'plan.json'),'sha256':sha(slot/'plan.json'),'review_sha256':sha(REVIEW),'phase_spec_sha256':sha(slot/'phase_spec.json'),'gate':d['blocking_gates']})
dump(OWN/'delivery.json',{'plans':summary,'helper':{'path':str(HELP),'sha256':sha(HELP)},'review':{'path':str(REVIEW),'sha256':sha(REVIEW)},'staging_script_sha256':sha(OWN/'stage_proposal.py'),'GPU_build_tests_executed':False,'shared_files_changed':[]})
print(json.dumps({'plans':len(summary),'review_sha256':sha(REVIEW),'raw_root':str(RAW)}))
