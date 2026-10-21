from pathlib import Path
import json,hashlib,datetime
O=Path(__file__).parent;D=O.with_name('profiler2025_rollout059_060');W=Path('/home/imo/HeCBench_julia');sha=lambda b:hashlib.sha256(b).hexdigest();H=lambda p:sha(Path(p).read_bytes());review=json.loads((D/'launch_review.json').read_bytes());diff=json.loads((D/'plan_diff_manifest.json').read_bytes());checks=[];issues=[]
def ck(name,passed,detail=None):
 checks.append({'check':name,'passed':bool(passed),'detail':detail})
 if not passed:issues.append({'check':name,'detail':detail})
def rp(x,a,b):
 if isinstance(x,str):return x.replace(a,b)
 if isinstance(x,list):return[rp(v,a,b)for v in x]
 if isinstance(x,dict):return{rp(k,a,b):rp(v,a,b)for k,v in x.items()}
 return x
ck('exact_seven_plans',len(diff['changes'])==7)
ck('exact_models',sorted((r['benchmark'],r['model'])for r in diff['changes'])==sorted([('miniDGS',m)for m in ['cuda','cuda_jl','ka','jacc']]+[('atomicSystemWide',m)for m in ['cuda_jl','ka','jacc']]))
ck('readonly_binding_count',len(review['readonly_bindings'])==600)
for path,h in review['readonly_bindings'].items():ck('readonly_file_binding',Path(path).is_file()and H(path)==h,{'path':path,'sha256':h})
for path,h in review['future_collector_tool_bindings'].items():ck('actual_adopted_tool_binding',Path(path).is_file()and H(path)==h,{'path':path,'sha256':h})
method=Path(review['method_review']['path']);ck('actual_canonical_review',H(method)==review['method_review']['sha256']=='65145043726f06195ef78dc89e72ca996b8fbb9217e4f2b8a3226e24e93b5eca');md=json.loads(method.read_bytes());bundle=method.parent/md['bundle_file'];ck('actual_bundle',H(bundle)==md['bundle_sha256']==review['method_review']['bundle_sha256']);ck('exact_profiler_metadata',md['profiler']==review['profiler']);ck('actual_review_approved',md['status']=='root_approved_after_saved_calibration_pass')
for pk,hk in [('executable','executable_sha256'),('target_ELF','target_ELF_sha256')]:ck('actual_profiler_file',H(md['profiler'][pk])==md['profiler'][hk],{'path':md['profiler'][pk]})
ck('controller_hash',H(D/'wait_then_collect.py')==review['controller_sha256']);future=json.loads((D/'future_argv.json').read_bytes())['argv'];ck('future_review_pin',future[future.index('--review-sha256')+1]==H(D/'launch_review.json'))
planchecks=[]
for row in diff['changes']:
 op=Path(row['parent']);np=Path(row['new_plan']);old=json.loads(op.read_bytes());new=json.loads(np.read_bytes());a,b=old['scratch_cwd'],new['scratch_cwd'];bench,m=row['benchmark'],row['model'];tag=bench+'/'+m
 ck('parent_plan_hash',H(op)==row['parent_sha256'],tag)
 for key in ['configuration_id','raw_config_sha256','reference_result_sha256','reference_result','runtime_assets','runtime_asset_bytes','source_cwd_readonly_inputs','effective_original_timing_region','profiler_requirements','blocking_gates']:
  ck('unchanged_selected_contract_'+key,new.get(key)==old.get(key),tag)
 for key in ['build_command','build_commands','profile_target_command','environment_overrides']:
  ck('path_only_command_'+key,rp(new.get(key),b,a)==old.get(key),tag)
 oldbindings=old['execution_bindings'];back=rp(new['execution_bindings'],b,a);extra=set(back)-set(oldbindings);ck('one_extra_exact_absolute_Julia_binding',extra=={'/home/imo/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia'}or extra==set(),{'slot':tag,'extras':sorted(extra)})
 for change in row['patches']:
  raw=change['original_text'].encode();out=change['text'].encode();ck('retained_original_copy_hashes',sha(raw)==change['original_sha256']and sha(out)==change['copy_sha256']==H(change['copy']),tag)
  src=Path(change['original']);ck('available_original_or_exact_named_recovery',((src.is_file()and H(src)==sha(raw))or(bench=='atomicSystemWide'and m=='ka'and not src.exists()and H(change['identical_current_byte_source'])==sha(raw)=='e092079d3302e79acb94c82f770c4fe6d716bf0f439f88656535cbd4578adb6f')),tag)
  if bench=='miniDGS'and m!='cuda':
   before=b"rankcmd=['julia','--startup-file=no'";after=b"rankcmd=['/home/imo/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia','--startup-file=no'";ck('exact_unique_nested_executable_only_transform',raw.count(before)==1 and raw.replace(before,after,1)==out and out.replace(after,before,1)==raw,tag)
  else:ck('copied_body_byte_identical',raw==out,tag)
  back[change['original']]=change['original_sha256']
 for k,v in oldbindings.items():ck('inverse_execution_binding',back.get(k)==v,{'slot':tag,'path':k})
 if bench=='atomicSystemWide':
  link=new['application_replay_review'];ck('atomic_new_review_hash',H(link['path'])==link['sha256'],tag);doc=json.loads(Path(link['path']).read_bytes())['models'][link['entry']];ck('atomic_review_source_true_runtime_pending',doc['source_reviewed']is True and doc['runtime_validated']is False,tag)
 else:ck('conditional_replay_preserved',all(new['conditional_replay_review'].get(k)==v for k,v in old['conditional_replay_review'].items()if k!='runtime_transport_validation'),tag)
 for key in set(old)|set(new):
  allowed={'scratch_cwd','build_command','build_commands','profile_target_command','environment_overrides','execution_bindings','new_profiler_rollout_provenance','application_replay_review','conditional_replay_review','staging_status','commands_executed'}
  if key not in allowed:ck('other_plan_field_unchanged',new.get(key)==old.get(key),{'slot':tag,'field':key})
 planchecks.append({'slot':tag,'parent_plan':{'path':str(op),'sha256':H(op)},'new_plan':{'path':str(np),'sha256':H(np)},'configuration_id':new['configuration_id'],'reference_result_sha256':new['reference_result_sha256'],'raw_config_sha256':new['raw_config_sha256'],'scratch_root':b,'copied_sources':len(row['patches'])})
sel=Path('/tmp/hecbench-roofline-v100-20261020/batch059-miniDGS-ncu2025/process_filter_selector.json');oldsel=Path('/tmp/hecbench-roofline-v100-20261020/batch040-miniDGS/process_filter_selector.json');ck('selector_semantic_identity',json.loads(sel.read_bytes())==json.loads(oldsel.read_bytes()));se=json.loads(sel.read_bytes());ck('no_expected_launch_count_inference',se['expected_matching_launches']is None)
for m,args in se['selection'].items():ck('exact_process_filter',args==['--filter-mode','per-launch-config','--kernel-name-base','mangled','--kernel-id','::regex:.*:1','--target-processes-filter','native_rank'if m=='cuda'else'julia'],m)
ck('exact_two_sequential_jobs',[(j['benchmark'],j['maximum_invocations'])for j in review['jobs']]==[('miniDGS',1),('atomicSystemWide',1)])
for j in review['jobs']:
 argv=j['argv'];tag=j['benchmark'];ck('explicit_skip_retain_method',all(k in argv for k in ['--skip-qualified-selected','--retain-auxiliary','--profiler-executable','--profiler-method-review','--profiler-method-review-sha256']),tag);ck('exact_timeout_and_reserve',argv[argv.index('--timeout')+1]==('900'if tag=='miniDGS'else'1800')and argv[argv.index('--reserve-mib')+1]=='2048',tag);ck('application_replay_flag_only_atomic',('--application-replay-reviewed'in argv)==(tag=='atomicSystemWide'),tag)
ck('exact_predecessor_set',[(x['pid'],x['start_ticks'])for x in review['predecessors']]==[(1227026,'76540052'),(1269349,'76747395'),(1451256,'77590853')])
# Source inspection only; never import/execute controller.
controller=(D/'wait_then_collect.py').read_text();ck('one_shot_existing_receipt_refusal',"need(not receipt.exists()"in controller);ck('bounded_predecessor_arg',"1<=a.maximum_predecessor_wait_seconds<=86400"in controller);ck('one_sequential_for_jobs_no_retry_loop',controller.count("for case in review['jobs']:")==1);ck('post_child_and_pre_child_source_deployment_checks',controller.count("deployed(review);verify(review['readonly_bindings'])")>=3);ck('no_signal_or_GPUquery_api',all(t not in controller for t in ['os.kill(', 'nvidia-smi', 'killpg(', 'sudo', 'signal.']));ck('current_controller_not_already_launched',(not(D/'controller_receipt.json').exists()))
report={'schema_version':1,'reviewed_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'root_launch_admissible':not issues,'blockers':issues,'checks_total':len(checks),'checks_passed':sum(x['passed']for x in checks),'checks':checks,'plans':planchecks,'packet_bindings':{str(D/n):H(D/n)for n in ['launch_review.json','wait_then_collect.py','plan_diff_manifest.json','atomic_application_replay_review.json','future_argv.json','plan_inventory.json']},'qualification':['Source/runtime admission only: new2025 method compatibility of actualMPI/applicationreplay remains unproved until runtime.','No successful futureGPUcapture claimed; oldfailed2022 captures untouched.','Allfive original checks required eachapplicationpass/two-rank run; atomicproperty is not identicalrawstate.','Source first-per-functiongeometry selection is partial; samplephase and fullROI remainunknown.','One finite futurefollower can be added as41st only after root actualPID/start/command review; current40frontierunchanged.','Predecessor wait bound86400s; collector foreignidlewaiting outside executiontimeout may be unbounded; rootfinite supervision remains necessary.','Absent old043KAsource recovered only from exacte092 bytebound retainedhelper into NEWroot, no historicalpath recreation.'],'full_five_checks_source_review':{'miniDGS_driver':'range(1,6) exacttwo-rank JSONcorrect/prefix/outputcount; fullbinarysize andconcat; ranknative0:5 checksfullstate/oracles/redzones eachretainedtrial, same mathsource pins.','atomic_pass_wrapper':'Require rowtrial[1,2,3,4,5],correct,checked42,maxabs0,full10rawfields+rangechecks,55wordexactpropertyartifact+32guards; no passcount inference.'},'no_GPU_build_tests_import_signal_runs':True,'shared_mutations':False}
q=O/'independent_review.json';q.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'root_launch_admissible':not issues,'checks':len(checks),'issues':issues,'review_sha256':H(q)}))
