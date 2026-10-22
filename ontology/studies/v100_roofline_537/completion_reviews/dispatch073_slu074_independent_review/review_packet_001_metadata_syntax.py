from pathlib import Path
import json,gzip,hashlib,fcntl,copy
O=Path(__file__).parent;P=O.with_name('dispatch073_slu074_finite_proposals');W=Path('/home/imo/HeCBench_julia');S=W/'ontology/studies/v100_roofline_537';C={};checks=[]
def H(p):
 p=Path(p);s=p.stat();k=(str(p),s.st_size,s.st_mtime_ns)
 if k not in C:
  h=hashlib.sha256()
  with p.open('rb')as f:
   for b in iter(lambda:f.read(4<<20),b''):h.update(b)
  C[k]=h.hexdigest()
 return C[k]
def ck(n,v,d=None):
 checks.append({'check':n,'passed':bool(v),'detail':d})
 if not v:raise ValueError((n,d))
def load(p):p=Path(p);return json.loads(gzip.decompress(p.read_bytes())if p.suffix=='.gz'else p.read_bytes())
def move(x,a,b):
 if isinstance(x,str):return x.replace(a,b)
 if isinstance(x,list):return [move(v,a,b)for v in x]
 if isinstance(x,dict):return {k:move(v,a,b)for k,v in x.items()}
 return x
ck('controller_exact',H(P/'controller.py')=='7d2e5117f24c2dfc9b6d64ecc0be68b1d22a6492a2cc34a33d96a01b745a26c1');ck('review_exact',H(P/'launch_review.json')=='f734d6c8e74467b8242d98f5bae3a1662bc153c7ba106d84312a2b9cc45223b4');r=load(P/'launch_review.json');ck('exact_jobs',{j['job_id']for j in r['jobs']}=={'slu__ka'}|{'dispatch__'+m for m in ['cuda','cuda_jl','ka','jacc']})
with open('/tmp/hecbench-roofline-metadata.lock','r')as lock:
 fcntl.flock(lock,fcntl.LOCK_SH)
 for typ in ['tool_bindings','packet_bindings']:
  for path,h in r[typ].items():ck('frozen_'+typ,H(path)==h,path)
 bproof=load(P/'build_manifest_inverse_proof.json');raw=gzip.decompress(Path(bproof['retained_compressed']['path']).read_bytes());ck('original_full_manifest_exact',raw==Path(bproof['original_full_bytes']['path']).read_bytes()and hashlib.sha256(raw).hexdigest()==bproof['original_captured_binding_SHA']);orig=json.loads(raw);freshbytes=Path(bproof['fresh_full_bytes']['path']).read_bytes();fresh=json.loads(freshbytes);ck('exact42commands91bindings',len(fresh['commands'])==42 and len(fresh['input_bindings'])==91)
 slujob=next(j for j in r['jobs']if j['benchmark']=='slu');oldscratch=slujob['old_scratch'];newscratch=slujob['scratch'];ck('manifest_complete_byte_inverse',freshbytes==raw.replace(oldscratch.encode(),newscratch.encode())and freshbytes.replace(newscratch.encode(),oldscratch.encode())==raw)
 for path,h in fresh['input_bindings'].items():ck('manifest_source_binding',H(path)==h,path)
 for cmd in fresh['commands']:
  dest=cmd[2]if Path(cmd[0]).name=='ar'else cmd[cmd.index('-o')+1];ck('compiler_output_fresh_only',Path(dest).is_relative_to(newscratch),dest)
 for typ in ['final_outputs','must_be_absent_before_build']:
  ck('all_declared_outputs_fresh',all(Path(x).is_relative_to(newscratch)for x in fresh[typ]),typ)
 oldcounts={}
 for j in r['jobs']:
  tag=j['job_id'];old=load(j['original_plan']['path']);new=load(j['prepared_plan']['path']);ck('prepared_SHA',H(j['prepared_plan']['path'])==j['prepared_plan']['sha256'],tag);ck('oldplan_matches_frozen',old==load(W/j['expected_latest_profile_id']/'plan.json.gz'),tag);l=load(S/'benchmarks'/f"{j['benchmark']}.json");ct=l['accepted_named_contract'];ck('current_contract',all(ct[k]==v for k,v in j['expected_contract'].items()),tag);ck('current_latest_capture',l['models'][j['model']]['latest_profile_attempt']['profile_id']==j['expected_latest_profile_id'],tag)
  for path,h in j['readonly_bindings'].items():ck('readonly_bound',H(path)==h,path)
  inv=copy.deepcopy(new)
  if j['benchmark']=='slu':
   oldmp=old['build_command'][old['build_command'].index('--build-manifest')+1];newmp=new['build_command'][new['build_command'].index('--build-manifest')+1];oldsha=old['execution_bindings'][oldmp];ck('old_manifest_capture_SHA',oldsha==hashlib.sha256(raw).hexdigest());ck('fresh_manifest_SHA_arg',new['build_command'][-1]==H(bproof['fresh_full_bytes']['path'])==new['execution_bindings'][newmp]);ck('one_exact_stage_artifact',len(j['stage_artifacts'])==1 and j['stage_artifacts'][0]['future_copy']==newmp and H(j['stage_artifacts'][0]['frozen_copy']['path'])==new['execution_bindings'][newmp]);del inv['execution_bindings'][newmp];inv['execution_bindings'][oldmp]=oldsha;inv['build_command']=move(inv['build_command'],newmp,oldmp);inv['build_command'][-1]=oldsha
  for k in ['scratch_cwd','environment_overrides','build_command','profile_target_command','runtime_assets']:inv[k]=move(inv[k],j['scratch'],j['old_scratch'])
  ck('complete_plan_inverse',inv==old,tag);ck('source_copies_none',not j['source_copies'],tag);ck('future_flat_root',j['plan_root']=='/tmp/hecbench-roofline-v100-20261020/'+('batch073'if j['benchmark']=='dispatch'else'batch074'));ck('rawroot_not_reused',not Path(j['plan_root']).exists());ck('GPU6_numeric_plan',new['environment_overrides']['CUDA_VISIBLE_DEVICES']=='6');ck('explicit_bounded_budget',j['timeout_seconds']==(3600 if j['benchmark']=='dispatch'else900) and j['argv'][j['argv'].index('--timeout')+1]==str(j['timeout_seconds']));ck('cleanup_skip','--retain-auxiliary'not in j['argv']and'--skip-qualified-selected'in j['argv'])
  pv=load(W/j['expected_latest_profile_id']/'profile_provenance.json.gz');logp=W/j['expected_latest_profile_id']/'execution.log.gz';log=gzip.decompress(logp.read_bytes());ck('retained_log_provenance_SHA',hashlib.sha256(log).hexdigest()==pv['profile']['log_sha256']);tr=[json.loads(z[12:])for z in log.decode().splitlines()if z.startswith('STUDY_TRIAL ')];oldcounts[tag]=len(tr);ck('all_old_complete_rows_correct',all(z['correct']is True for z in tr));oldselector=load(pv['selection_override']['path']);sel=load(j['selection']['path']);ck('exact_selector_preserved',sel==oldselector)
  if j['benchmark']=='dispatch':ck('execution_only_oldrows',all(z['checked_values']==0 and z['source_empty_kernel']is True and z['execution_checked']is True and z['completed_source_loop_launch_calls']==1102200 for z in tr));ck('all4_selector',set(sel['selection'])=={'cuda','cuda_jl','ka','jacc'})
  else:ck('four_actual_full_SLU_rows',len(tr)==4 and [z['trial']for z in tr]==[1,2,3,4]and all(z['checked_values']==16784916 and z['original_CPUpreprocess_symbolic_and_solve_execute_outside_ROI']is True for z in tr));ck('full_warmup_observed','FullSLU warmup correct=1'in log.decode());ck('full_ASIC_case_pass_5times',log.decode().count('SLU independent case=0 n=99340 pass=1')>=5)
 fcntl.flock(lock,fcntl.LOCK_UN)
source=load(P/'job_inventory.json')['source_proofs']
for q in source:ck('frozen_source_exact',H(q['original'])==q['original_sha256']==H(q['frozen']['path']),q['original'])
report={'passed_source_review':True,'root_admissible_subject_to_final_atomic_revalidation':True,'checks_total':len(checks),'checks':checks,'actual_saved_complete_rows':oldcounts,'build_manifest_commands':42,'build_manifest_input_bindings':91,'runtime_pass_not_claimed':True,'GPU_build_tests_candidate_imports_sharededits':0,'qualifications':['Onefinitecall/group, no retry; dispatch4models traversed serialinsideoneCLI, foreignhalt observed onlyafterCLIreturns, eachmodel independentlyGPU6idle/lock/guarded.','All5freshactualchecks+fullSLU4cases/beforeafterworkspace manifests required; old4doesnotcountasnewsuccess.','Dispatch empty-launch execution-only, not numerical or FPwork.','SLU99340ASIC/full4cases preserved, host symbolic/oracle/workspaceaudits remain outsidekernelROI; no alteredscientificthreshold/math.','Scratchcleanup onlyafterfullmanifest; freshbinarynumericalrecertification unavailableaftercleanup.','Process900/3600budget explicit, not causaldiagnosis or runtimeguarantee.']};(O/'independent_review.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'checks':len(checks),'report_SHA':H(O/'independent_review.json'),'oldcounts':oldcounts}))
