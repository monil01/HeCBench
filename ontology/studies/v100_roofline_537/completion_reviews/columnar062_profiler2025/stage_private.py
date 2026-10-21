from pathlib import Path
import json,hashlib,copy,os
D=Path(__file__).resolve().parent;R=Path('/home/imo/HeCBench_julia');RAW=Path('/tmp/hecbench-roofline-v100-20261020');PARENT=RAW/'batch021-retry-columnarSolver';OUT=RAW/'batch062-columnarSolver-ncu2025';M=R/'ontology/studies/v100_roofline_537/profiler_methods/ncu2025_2_0_V100_sm70_raw_v1/review.json';MSHA='65145043726f06195ef78dc89e72ca996b8fbb9217e4f2b8a3226e24e93b5eca';J=Path('/home/imo/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def need(ok,msg):
 if not ok:raise ValueError(msg)
def dump(p,v):Path(p).write_text(json.dumps(v,indent=2)+'\n')
def remap(x,a,b):
 if isinstance(x,str):return x.replace(a,b)
 if isinstance(x,list):return[remap(z,a,b)for z in x]
 if isinstance(x,dict):return{remap(k,a,b):remap(v,a,b)for k,v in x.items()}
 return x
need(not OUT.exists(),'Fresh062required');need(M.is_file()and sha(M)==MSHA,'Actualcanonicalmethodreviewrequired');method=json.loads(M.read_bytes());bundle=M.parent/method['bundle_file'];need(sha(bundle)==method['bundle_sha256'],'Methodbundleidentity')
bindings={str(M):MSHA,str(bundle):sha(bundle)};changes=[];inventory=[];OUT.mkdir()
selector=PARENT/'bounded_selector.json';selraw=selector.read_bytes();sel=json.loads(selraw)
for m in ['ka','jacc']:need(sel['selection'][m]==['--filter-mode','global','--kernel-name-base','mangled','--kernel-id','::regex:.*:1'],'Exactparentglobalfirstselector')
# Keep the whole selector byte-for-byte, including historicalCUDAjl entry:
# only the two freshmodelplan directories are traversed by collector.
(OUT/'bounded_selector.json').write_bytes(selraw);bindings[str(OUT/'bounded_selector.json')]=sha(OUT/'bounded_selector.json');bindings[str(selector)]=sha(selector)
for m in ['ka','jacc']:
 pp=PARENT/'columnarSolver'/m/'plan.json';original=json.loads(pp.read_bytes());p=copy.deepcopy(original);slot=OUT/'columnarSolver'/m;scratch=slot/'scratch';(scratch/'profile_outputs').mkdir(parents=True);(scratch/'ncu_tmp').mkdir();(scratch/'private_trace').mkdir();old=original['scratch_cwd'];new=str(scratch)
 for k in ['scratch_cwd','build_command','profile_target_command','environment_overrides','execution_bindings','runtime_assets']:
  if k in p:p[k]=remap(p[k],old,new)
 need(p['build_command']==[],'No build permitted bycopiedtargetplan');need(remap(p['profile_target_command'],new,old)==original['profile_target_command'],'Exactinversecommandbytes');need(p['configuration_id']==sel['configuration_id'],'SelectorCID')
 p['execution_bindings'][str(J)]=sha(J);p['execution_bindings'][str(OUT/'bounded_selector.json')]=sha(OUT/'bounded_selector.json')
 for path,h in p['execution_bindings'].items():need(Path(path).is_file()and sha(path)==h,'Actualsourcefixturebindingdrift '+path);bindings[path]=h
 prov=PARENT/'columnarSolver'/m/'profile_provenance.json';pr=json.loads(prov.read_bytes());need(pr['profile']['stop_reason']=='wall_timeout'and pr['profile']['returncode']!=0,'Actual021timeoutparentrequired')
 bindings[str(pp)]=sha(pp);bindings[str(prov)]=sha(prov);p['profiler_method_rollout']={'root_review_required':True,'parent_plan':str(pp),'parent_plan_sha256':sha(pp),'parent_failed_provenance':str(prov),'parent_failed_provenance_sha256':sha(prov),'method_review':{'path':str(M),'sha256':MSHA},'source_bytes_changed':False,'target_command_scratch_substitution_exact_inverse':True,'parent_configuration_and_scientific_contract_unchanged':True,'effective_profile_deadline_request':3600,'unknown_timeout_cause':True,'no_claim_newSDK_repairs_target':True,'counter_scope':'Exactoriginalglobalfirstselector; partialcapturedkernelset,geometry/state/phaseundercoverage; notcompleteROI/causal','root_primary_times_unchanged':True}
 p['staging_status']='private_source_staged_root_review_required_062';p['commands_executed']=False;dump(slot/'plan.json',p);bindings[str(slot/'plan.json')]=sha(slot/'plan.json');inventory.append({'model':m,'path':str(slot/'plan.json'),'sha256':sha(slot/'plan.json'),'configuration_id':p['configuration_id'],'raw_config_sha256':p['raw_config_sha256'],'reference_result_sha256':p['reference_result_sha256']});changes.append({'model':m,'original_plan':str(pp),'original_sha256':sha(pp),'fresh_scratch':new,'old_scratch':old,'kernel_source_edits':0,'target_command_before':original['profile_target_command'],'target_command_after':p['profile_target_command'],'exact_inverse_scratch_path_substitution':True,'new_bound_actualJuliaELF':{'path':str(J),'sha256':sha(J)},'actual_parent_deadline':pr['operational_deadline'],'actual_parent_wall_seconds':pr['profile']['wall_seconds']})
for path,h in bindings.items():need(sha(path)==h,'Beforeafterbindingdrift')
pred={'pid':1141998,'start_ticks':'76117190','uid':os.getuid()};proc=Path('/proc/1141998')
try:
 st=(proc/'stat').read_text().rsplit(')',1)[1].split();pred['observed_same_generation']=st[19]==pred['start_ticks'];pred['observed_state']=st[0];pred['argv']=[x.decode()for x in(proc/'cmdline').read_bytes().split(b'\0')if x]if pred['observed_same_generation']else None
except(FileNotFoundError,ProcessLookupError):pred.update(observed_same_generation=False,observed_state='absent',argv=None)
future={str(R/'tools'/n):sha(R/'tools'/n)for n in ['v100_roofline_collect.py','v100_profiler_method.py','v100_roofline_classify.py','v100_profile_write_guard.py']};need(future[str(R/'tools/v100_roofline_collect.py')]=='c21b1edd4c6e0924250b306e1b97ff7ff1371b8407eb92899e089dbdca364116','ExactadoptedC21collector')
argv=['/usr/bin/python3',str(R/'tools/v100_roofline_collect.py'),'--source-root',str(R),'--plan-root',str(OUT),'--timeout','3600','--reserve-mib','2048','--retain-auxiliary','--skip-qualified-selected','--selection-overrides',str(OUT/'bounded_selector.json'),'--profiler-executable',method['profiler']['executable'],'--profiler-method-review',str(M),'--profiler-method-review-sha256',MSHA]
review={'schema_version':1,'status':'PRIVATE_ROOT_REVIEW_REQUIRED_NOT_LAUNCHED','source_root':str(R),'future_collector_tool_bindings':future,'method_review':{'path':str(M),'sha256':MSHA,'bundle_sha256':method['bundle_sha256']},'profiler':method['profiler'],'readonly_bindings':bindings,'predecessors':[pred],'jobs':[{'benchmark':'columnarSolver','argv':argv,'maximum_invocations':1}],'maximum_collector_invocations':1,'maximum_predecessor_wait_seconds':86400,'source_reviewed_timeoutqualification':'Causeunknown; fullsame5checksandallscientificbuffers preserved. Newversion/3600deadline explicitoperationalexperiment, notproofSDKfix.'};dump(D/'launch_review.json',review);dump(D/'plan_inventory.json',inventory);dump(D/'source_path_inverse_review.json',{'changes':changes,'source_edits':0,'source_copies_required':False,'byte_identical_selector':{'original':str(selector),'copy':str(OUT/'bounded_selector.json'),'sha256':sha(selector)},'source_checked_contract':{'full_source397cipher','key30','150000climbings_per1024trajectories'} if False else['fixedsupplied397cipher/key30/150000x1024','same warmup+fivecheckedfullprogram executions','samewhole-device setup+decode timer/sourceCPUexactstate+independentkeyplaintext/bigram/permutation/knownfixturewitness','sameall676Float32scores/397Int32cipher actualreadbacks/fullstate/guards; noexternal target/input'],'no_newacceptance':True});print(json.dumps({'plans':inventory,'predecessor':pred,'method':MSHA,'tools':future}))
