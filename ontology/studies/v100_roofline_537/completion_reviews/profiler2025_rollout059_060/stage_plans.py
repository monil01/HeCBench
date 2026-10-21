from pathlib import Path
import json,hashlib,copy,difflib,os
D=Path(__file__).resolve().parent;RAW=Path('/tmp/hecbench-roofline-v100-20261020');ROOT=Path('/home/imo/HeCBench_julia');J='/home/imo/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia';METHOD_SHA='65145043726f06195ef78dc89e72ca996b8fbb9217e4f2b8a3226e24e93b5eca';METHOD=ROOT/'ontology/studies/v100_roofline_537/profiler_methods/ncu2025_2_0_V100_sm70_raw_v1/review.json';CAND=Path('/tmp/roofline-completion-20261021/profiler2025_method_patch/v5/proposed/tools');PARENTS={'miniDGS':RAW/'batch040-miniDGS','atomicSystemWide':RAW/'batch057-atomic-app-recovery'};OUT={'miniDGS':RAW/'batch059-miniDGS-ncu2025','atomicSystemWide':RAW/'batch060-atomic-app-ncu2025'}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):Path(p).write_text(json.dumps(v,indent=2)+'\n')
def need(ok,msg):
 if not ok:raise ValueError(msg)
def replace_paths(x,old,new):
 if isinstance(x,str):return x.replace(old,new)
 if isinstance(x,list):return[replace_paths(v,old,new)for v in x]
 if isinstance(x,dict):return{replace_paths(k,old,new):replace_paths(v,old,new)for k,v in x.items()}
 return x
for p in OUT.values():need(not p.exists(),'Fresh root-assigned batch required '+str(p))
private_method=Path('/tmp/roofline-completion-20261021/profiler2025_actual003_admitted_bundle/review.json');need(sha(private_method)==METHOD_SHA,'Root-supplied actualmethodreview SHA');method=json.loads(private_method.read_bytes());need(method['status']=='root_approved_after_saved_calibration_pass','No actualreviewapproval')
review_models={};plans=[];allbindings={};origparents={};changes=[]
for bench,models in [('miniDGS',['cuda','cuda_jl','ka','jacc']),('atomicSystemWide',['cuda_jl','ka','jacc'])]:
 for m in models:
  parent=(RAW/'batch043-atomic-app-replay'if bench=='atomicSystemWide'and m=='ka'else PARENTS[bench])/bench/m/'plan.json';original=json.loads(parent.read_bytes());p=copy.deepcopy(original);slot=OUT[bench]/bench/m;scratch=slot/'scratch';(scratch/'profile_source').mkdir(parents=True);(scratch/'profile_outputs').mkdir();(scratch/'ncu_tmp').mkdir();old=original['scratch_cwd'];new=str(scratch)
  for field in ['scratch_cwd','build_command','build_commands','profile_target_command','environment_overrides','execution_bindings']:
   if field in p:p[field]=replace_paths(p[field],old,new)
  patchlist=[]
  for oldpath,h in original['execution_bindings'].items():
   if not oldpath.startswith(old+'/'):continue
   op=Path(oldpath);dest=Path(oldpath.replace(old,new,1));byte_source=op
   if not op.exists():
    need(bench=='atomicSystemWide'and m=='ka'and op.name=='atomic_application_pass.py'and h=='e092079d3302e79acb94c82f770c4fe6d716bf0f439f88656535cbd4578adb6f','Unavailable source has no narrowly reviewed byte recovery')
    byte_source=Path('/tmp/roofline-completion-20261021/remaining_replay/atomic_application_pass.py')
   need(byte_source.is_file()and sha(byte_source)==h,'Active copied source bytes absent/changed '+str(byte_source));dest.parent.mkdir(parents=True,exist_ok=True);raw=byte_source.read_bytes();out=raw
   if bench=='miniDGS'and m!='cuda'and op.name=='study_driver.py':
    token=b"rankcmd=['julia','--startup-file=no'";revised=b"rankcmd=['"+J.encode()+b"','--startup-file=no'";need(raw.count(token)==1,'Nested Julia unique');out=raw.replace(token,revised,1);need(out.replace(revised,token,1)==raw,'Inverse copy identity')
   dest.write_bytes(out);p['execution_bindings'][str(dest)]=sha(dest);patchlist.append({'original':str(op),'original_sha256':h,'identical_current_byte_source':str(byte_source),'historical_source_recreated':False,'copy':str(dest),'copy_sha256':sha(dest),'original_text':raw.decode(),'text':out.decode(),'unified_diff':list(difflib.unified_diff(raw.decode().splitlines(True),out.decode().splitlines(True),fromfile=str(op),tofile=str(dest))),'inverse_byte_equal':out==raw if bench!='miniDGS'or m=='cuda'else out.replace(revised,token,1)==raw})
  p['execution_bindings'][J]=sha(J)
  p['new_profiler_rollout_provenance']={'parent_plan':str(parent),'parent_plan_sha256':sha(parent),'root_review_required':True,'method_review_planned_path':str(METHOD),'method_review_sha256':METHOD_SHA,'new_method_not_yet_deployed_at_authoring':True,'profile_only_copied_source_patches':patchlist,'numeric_source_kernel_checks_unchanged':True,'original_config_sha256':p['raw_config_sha256'],'original_result_sha256':p['reference_result_sha256'],'counter_scope':'Partial first matching invocation per recorded wildcard/per-launch-config filters; not complete family/ROI inventory.','parent_attempts_retained_unchanged':True}
  need(p['configuration_id']==original['configuration_id']and p['raw_config_sha256']==original['raw_config_sha256']and p['reference_result_sha256']==original['reference_result_sha256'],'Identitychanged')
  if bench=='atomicSystemWide':
   link=original['application_replay_review'];doc=json.loads(Path(link['path']).read_bytes());need(sha(link['path'])==link['sha256'],'Parentatomicreview');rec=copy.deepcopy(doc['models'][link['entry']]);rec['execution_bindings']=replace_paths(rec['execution_bindings'],old,new)
   # All new active source bindings must match the fresh plan, old frozen refs
   # remain readonly historical evidence, not the new collector executable.
   for k in list(rec['execution_bindings']):
    if k in p['execution_bindings']:rec['execution_bindings'][k]=p['execution_bindings'][k]
   rec['rollout_parent_review']={'path':link['path'],'sha256':link['sha256'],'entry':link['entry']};rec['method_version_qualification']='Same source-reviewed application/grid/strict replay and all5 full property checks; fresh2025 method compatibility remains actualruntime observation.';rec['source_reviewed']=True;rec['runtime_validated']=False;review_models[link['entry']]=rec
   p['application_replay_review']={'path':str(D/'atomic_application_replay_review.json'),'sha256':'AUTHORED_AFTER_ALL_RECORDS','entry':link['entry']}
  else:p['conditional_replay_review']['runtime_transport_validation']='Pending fresh actual2025 observation; original MPI2/GPU6 and finite rank-local kernels preserved.'
  p['staging_status']='source_staged_requires_root_review_newmethod_deployment_and_predecessor_completion';p['commands_executed']=False;plans.append((slot/'plan.json',p));origparents[str(parent)]=sha(parent);changes.append({'benchmark':bench,'model':m,'parent':str(parent),'parent_sha256':sha(parent),'new_plan':str(slot/'plan.json'),'patches':patchlist,'build_command_preserved_under_scratch_relocation':replace_paths(p.get('build_command',[]),new,old)==original.get('build_command',[])})
if review_models:dump(D/'atomic_application_replay_review.json',{'schema_version':1,'models':review_models,'scope':'Prospective freshpaths/new2025 compatibility; root CPU/source review required before launch, no GPU execution performed.'})
selector=json.loads((RAW/'batch040-miniDGS/process_filter_selector.json').read_bytes());dump(OUT['miniDGS']/'process_filter_selector.json',selector)
for path,p in plans:
 if p['benchmark']=='atomicSystemWide':p['application_replay_review']['sha256']=sha(D/'atomic_application_replay_review.json')
 dump(path,p)
 for k,h in p['execution_bindings'].items():
  need(Path(k).is_file()and sha(k)==h,'Actualsourcebindingdrift '+k);allbindings[k]=h
 allbindings[str(path)]=sha(path)
allbindings[str(D/'atomic_application_replay_review.json')]=sha(D/'atomic_application_replay_review.json');allbindings[str(OUT['miniDGS']/'process_filter_selector.json')]=sha(OUT['miniDGS']/'process_filter_selector.json');allbindings.update(origparents)
# Pin the reviewed FUTURE implementation, not live oldbytes. No adoption occurs.
tools={str(ROOT/'tools'/name):sha(CAND/name)for name in ['v100_roofline_collect.py','v100_profiler_method.py']};tools[str(ROOT/'tools/v100_roofline_classify.py')]=sha(ROOT/'tools/v100_roofline_classify.py');tools[str(ROOT/'tools/v100_profile_write_guard.py')]=sha(ROOT/'tools/v100_profile_write_guard.py')
allbindings[str(private_method)]=METHOD_SHA
jobs=[]
for bench,timeout in [('miniDGS',900),('atomicSystemWide',1800)]:
 argv=['/usr/bin/python3',str(ROOT/'tools/v100_roofline_collect.py'),'--source-root',str(ROOT),'--plan-root',str(OUT[bench]),'--timeout',str(timeout),'--reserve-mib','2048','--retain-auxiliary','--skip-qualified-selected','--profiler-executable',method['profiler']['executable'],'--profiler-method-review',str(METHOD),'--profiler-method-review-sha256',METHOD_SHA]
 if bench=='miniDGS':argv+=['--selection-overrides',str(OUT[bench]/'process_filter_selector.json')]
 else:argv+=['--application-replay-reviewed']
 jobs.append({'benchmark':bench,'argv':argv,'maximum_invocations':1})
pred=[{'pid':1227026,'start_ticks':'76540052','uid':os.getuid()},{'pid':1269349,'start_ticks':'76747395','uid':os.getuid()},{'pid':1451256,'start_ticks':'77590853','uid':os.getuid()}]
for x in pred:
 proc=Path('/proc')/str(x['pid'])
 try:
  st=(proc/'stat').read_text().rsplit(')',1)[1].split();same=st[19]==x['start_ticks'];x['observed_same_generation']=same;x['observed_state']=st[0];x['argv']=[b.decode()for b in(proc/'cmdline').read_bytes().split(b'\0')if b]if same else None
 except(FileNotFoundError,ProcessLookupError):x.update(observed_same_generation=False,observed_state='absent',argv=None)
launch={'schema_version':1,'status':'ROOT_REVIEW_REQUIRED_NOT_LAUNCHED','source_root':str(ROOT),'future_collector_tool_bindings':tools,'method_review':{'path':str(METHOD),'sha256':METHOD_SHA,'bundle_sha256':method['bundle_sha256']},'profiler':method['profiler'],'readonly_bindings':allbindings,'predecessors':pred,'jobs':jobs,'maximum_collector_invocations':2,'maximum_predecessor_wait_seconds':86400,'initial_selected_skip_policy':'Collector skip-qualified-selected mandatory; alreadyqualified nativeatomic omitted from060. All040/043/056-058 generations mustfinish before either freshjob.'}
dump(D/'launch_review.json',launch);dump(D/'plan_diff_manifest.json',{'parents':origparents,'changes':changes,'counts':{'miniDGS':4,'atomicSystemWide':3},'new_source_kernel_math':False,'builds_runs_imports_tests':0,'shared_metadata_mutations':0,'planned_method_review_not_installed_at_authoring':not METHOD.exists()});print(json.dumps({'plans':len(plans),'launch_review_sha256':sha(D/'launch_review.json'),'future_tools':tools,'predecessors':pred}))
