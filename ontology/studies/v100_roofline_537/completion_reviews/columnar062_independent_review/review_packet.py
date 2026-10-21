from pathlib import Path
import json,hashlib,datetime,os
O=Path(__file__).parent;D=O.with_name('columnar062_profiler2025');W=Path('/home/imo/HeCBench_julia');H=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();r=json.loads((D/'launch_review.json').read_bytes());inv=json.loads((D/'plan_inventory.json').read_bytes());diff=json.loads((D/'source_path_inverse_review.json').read_bytes());checks=[];issues=[]
def ck(name,passed,detail=None):
 checks.append({'check':name,'passed':bool(passed),'detail':detail})
 if not passed:issues.append({'check':name,'detail':detail})
def rp(x,a,b):
 if isinstance(x,str):return x.replace(a,b)
 if isinstance(x,list):return[rp(z,a,b)for z in x]
 if isinstance(x,dict):return{rp(k,a,b):rp(v,a,b)for k,v in x.items()}
 return x
for path,h in r['readonly_bindings'].items():ck('readonly_binding',Path(path).is_file()and H(path)==h,{'path':path,'sha256':h})
for path,h in r['future_collector_tool_bindings'].items():ck('actual_adopted_tool_binding',Path(path).is_file()and H(path)==h,{'path':path,'sha256':h})
ck('controller_source_pin',H(D/'wait_then_collect.py')==r['controller_sha256']);future=json.loads((D/'future_argv.json').read_bytes())['argv'];ck('future_argv_review_pin',future[future.index('--review-sha256')+1]==H(D/'launch_review.json'))
mp=Path(r['method_review']['path']);md=json.loads(mp.read_bytes());ck('canonical_method_review',H(mp)==r['method_review']['sha256']=='65145043726f06195ef78dc89e72ca996b8fbb9217e4f2b8a3226e24e93b5eca');ck('canonical_method_bundle',H(mp.parent/md['bundle_file'])==r['method_review']['bundle_sha256']==md['bundle_sha256']=='12ad8f0b5d87fa45f14cd6546f2f3302307cb2f25f1b24bf8023eb919783e685');ck('profiler_identity',md['profiler']==r['profiler']and md['status']=='root_approved_after_saved_calibration_pass')
for p,h in [('executable','executable_sha256'),('target_ELF','target_ELF_sha256')]:ck('actual_profiler_file',H(md['profiler'][p])==md['profiler'][h],p)
ck('exact_two_models',len(inv)==2 and{z['model']for z in inv}=={'ka','jacc'})
planjoins=[]
for row in inv:
 pp=Path(row['path']);n=json.loads(pp.read_bytes());pr=next(x for x in diff['changes']if x['model']==row['model']);oldp=Path(pr['original_plan']);old=json.loads(oldp.read_bytes());tag=row['model'];a,b=old['scratch_cwd'],n['scratch_cwd'];ck('newplan_hash',H(pp)==row['sha256'],tag);ck('oldplan_hash',H(oldp)==pr['original_sha256'],tag)
 for k in ['configuration_id','raw_config_sha256','reference_result_sha256','reference_result','runtime_asset_bytes','profiler_requirements','effective_original_timing_region','blocking_gates']:
  ck('selected_scientific_contract_unchanged',n.get(k)==old.get(k),{'model':tag,'key':k})
 ck('exact_CID',n['configuration_id']=='columnarSolver_fixed397_fullKEY30_150000x1024_parser_index_volatile_corrected_v2',tag)
 for k in ['scratch_cwd','build_command','profile_target_command','environment_overrides','runtime_assets']:
  ck('path_inverse_'+k,rp(n.get(k),b,a)==old.get(k),tag)
 ck('no_build',n['build_command']==[],tag)
 back=rp(n['execution_bindings'],b,a)
 for path,h in old['execution_bindings'].items():ck('source_execution_binding_preserved',back.get(path)==h,{'model':tag,'path':path})
 extras=set(n['execution_bindings'])-set(old['execution_bindings']);ck('only_two_new_pins',extras=={'/home/imo/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia','/tmp/hecbench-roofline-v100-20261020/batch062-columnarSolver-ncu2025/bounded_selector.json'},tag)
 for k in set(n)|set(old):
  if k not in ['scratch_cwd','build_command','profile_target_command','environment_overrides','execution_bindings','runtime_assets','profiler_method_rollout','staging_status','commands_executed']:ck('other_field_unchanged',n.get(k)==old.get(k),{'model':tag,'key':k})
 prov=n['profiler_method_rollout'];ck('original_timeout_provenance_pin',H(prov['parent_failed_provenance'])==prov['parent_failed_provenance_sha256'],tag);oldpr=json.loads(Path(prov['parent_failed_provenance']).read_bytes());ck('parent_really_failed_timeout',oldpr['profile']['stop_reason']=='wall_timeout'and oldpr['profile']['returncode']!=0,tag)
 fixture=Path('/tmp/columnarSolver_native/cpu_separate.bin');ck('full_CPU_fixture_bound',str(fixture)in n['execution_bindings']and fixture.stat().st_size==1888256,tag)
 planjoins.append({'model':tag,'old_plan':{'path':str(oldp),'sha256':H(oldp)},'new_plan':{'path':str(pp),'sha256':H(pp)},'CID':n['configuration_id'],'config_sha256':n['raw_config_sha256'],'result_sha256':n['reference_result_sha256']})
sel=Path('/tmp/hecbench-roofline-v100-20261020/batch062-columnarSolver-ncu2025/bounded_selector.json');oldsel=Path(diff['byte_identical_selector']['original']);ck('selector_exact_bytes',sel.read_bytes()==oldsel.read_bytes()and H(sel)=='3662f1f0f228b01ae1e53222c78a6db980ea48fbd22fb8d06b01b02f07cd2e2f');sd=json.loads(sel.read_bytes())
for m in ['ka','jacc']:ck('original_globalfirst_selector',sd['selection'][m]==['--filter-mode','global','--kernel-name-base','mangled','--kernel-id','::regex:.*:1'],m)
ck('fresh_only_KA_JACC_plan_dirs',sorted(p.parent.name for p in sel.parent.glob('columnarSolver/*/plan.json'))==['jacc','ka'])
ck('whole021_single_predecessor',len(r['predecessors'])==1 and r['predecessors'][0]['pid']==1141998 and r['predecessors'][0]['start_ticks']=='76117190')
e=r['predecessors'][0];p=Path('/proc/1141998');actual={'status':'absent'}
if p.exists():
 st=(p/'stat').read_text().rsplit(')',1)[1].split();actual={'start_ticks':st[19],'state':st[0],'uid':p.stat().st_uid,'argv':[x.decode()for x in(p/'cmdline').read_bytes().split(b'\0')if x]}
 if st[19]==e['start_ticks']and st[0]!='Z':ck('actual_whole021_owned_argv',actual['uid']==e['uid']==os.getuid()and actual['argv']==e['argv'])
ck('one_job_once',len(r['jobs'])==1 and r['maximum_collector_invocations']==1 and r['jobs'][0]['maximum_invocations']==1);argv=r['jobs'][0]['argv'];ck('exact_budget',argv[argv.index('--timeout')+1]=='3600'and argv[argv.index('--reserve-mib')+1]=='2048');ck('mandatory_retain_skip_method',all(x in argv for x in ['--retain-auxiliary','--skip-qualified-selected','--selection-overrides','--profiler-executable','--profiler-method-review','--profiler-method-review-sha256']));ck('only_fresh062_planroot',argv.count('--plan-root')==1 and argv[argv.index('--plan-root')+1]==str(sel.parent));ck('no_application_replay_unrequested',not any(x in argv for x in ['--application-replay-reviewed','--profile-window-reviewed','--isolated-assertion-reviewed']))
follower=(D/'wait_then_collect.py').read_text();ck('one_shot_receipt_refusal',"need(not receipt.exists()"in follower);ck('wait86400_bounded',"1<=a.maximum_predecessor_wait_seconds<=86400"in follower);ck('one_for_jobs_no_automatic_retry',follower.count("for case in review['jobs']:")==1);ck('no_signals_GPU_OS_actions',all(x not in follower for x in ['os.kill(', 'killpg(', 'nvidia-smi', 'sudo', 'signal.']));ck('deployment_binding_checks_before_and_after',follower.count("deployed(review);verify(review['readonly_bindings'])")>=3);ck('follower_not_launched',not(D/'controller_receipt.json').exists())
common=W/'src/columnarSolver-julia/study_common.jl';text=common.read_text();ck('full_warmup_five', 'for trial in 0:5'in text and 'trial>0 ? "STUDY_TRIAL " : "WARMUP "'in text);ck('full_raw_and_guards_and_independent_checks',all(t in text for t in ['raw==gold||error','independent(actualscores,actualcipher,out,diag)','guards=vcat(map(guardbytes,buffers)...)','launch_both!(gs,ge,gr,go,gd);CUDA.device_synchronize()']));ck('actual_inputbits_checks','actualscores==scores&&actualcipher==cipher'in text and 'Readonlyinputchanged'in text)
report={'schema_version':1,'reviewed_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'root_launch_admissible':not issues,'blockers':issues,'checks_total':len(checks),'checks_passed':sum(x['passed']for x in checks),'checks':checks,'planjoins':planjoins,'actual_predecessor_observation':actual,'packet_bindings':{str(D/n):H(D/n)for n in ['launch_review.json','wait_then_collect.py','source_path_inverse_review.json','future_argv.json','plan_inventory.json']},'scope':'One future experimental2025 operational attempt for KA/JACC fullfixed397fixture only. Explicit correctednative provenance unchanged, not untouchedsource/arbitraryworkload.','qualifications':['Whole021parent wait preserves everyoldqueueditem includingSLU; noCUDAjl058/native planroot or newmaster.','3600deadline is changedoperationalexperiment, notprovedSDKfix; prior walltimeouts preserved.','Sourcefixed30key/150000climbings/1024trajectories, setup+decodeROI andall5fullchecks unchanged.','Byteidenticalglobalfirst selector is partial sample, notfunctiongeometry/fullROI/actualphase proof.','Newmethodcompatibility unknownuntilactualrun; collectorreturncode alone no acceptedcapture.','Foreign/idlewait may beunbounded; rootfinite supervisor and actualnewPID/start/argv review required.','No GPUquery/run/build/test/targetimport/signals/OS/sharedmutation.'],'source_check_binding':{'path':str(common),'sha256':H(common)},'GPU_runs':0,'tests':0,'shared_mutations':False}
q=O/'independent_review.json';q.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'root_launch_admissible':not issues,'checks':len(checks),'issues':issues,'review_sha256':H(q)}))
