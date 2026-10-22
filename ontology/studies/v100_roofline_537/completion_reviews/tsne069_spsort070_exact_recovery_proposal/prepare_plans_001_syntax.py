from pathlib import Path
import json,gzip,hashlib,copy,fcntl,datetime
W=Path('/home/imo/HeCBench_julia');S=W/'ontology/studies/v100_roofline_537';O=Path(__file__).resolve().parent;RAW=Path('/tmp/hecbench-roofline-v100-20261020');sha_cache={}
def sha(p):
 p=Path(p);st=p.stat();key=(str(p),st.st_size,st.st_mtime_ns)
 if key not in sha_cache:
  h=hashlib.sha256()
  with p.open('rb')as f:
   for b in iter(lambda:f.read(4<<20),b''):h.update(b)
  sha_cache[key]=h.hexdigest()
 return sha_cache[key]
def info(p):return {'path':str(p),'sha256':sha(p),'bytes':Path(p).stat().st_size}
def load(p):p=Path(p);return json.loads(gzip.decompress(p.read_bytes())if p.suffix=='.gz'else p.read_bytes())
def move(x,old,new):
 if isinstance(x,str):return x.replace(old,new)
 if isinstance(x,list):return[move(y,old,new)for y in x]
 if isinstance(x,dict):return{k:move(v,old,new)for k,v in x.items()}
 return x
def save(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,indent=2)+'\n')
jobs=[];selectors=[];inverse=[];tproof=load(O/'tsne_exact_source_recovery_proof.json')
with Path('/tmp/hecbench-roofline-metadata.lock').open('r')as lk:
 fcntl.flock(lk,fcntl.LOCK_SH)
 for n,m in [('tsne','ka')]+[('spsort',m)for m in ['cuda','cuda_jl','ka','jacc']]:
  ledger=load(S/'benchmarks'/f'{n}.json');state=ledger['models'][m];ct=ledger['accepted_named_contract'];pid=state['latest_profile_attempt']['profile_id'];cap=W/pid;plan=load(cap/'plan.json.gz');prov=load(cap/'profile_provenance.json.gz');assert plan['configuration_id']==ct['configuration_id']and plan['raw_config_sha256']==ct['raw_config_sha256']and plan['reference_result_sha256']==ct['result_sha256'];assert state['profiling_status']!='profile_process_completed';assert sha(plan['reference_result'])==ct['result_sha256'];assert prov.get('profiler_method')is None;assert prov['profiler_executable']=='/usr/lib/nsight-compute/ncu';assert sha(prov['profiler_executable'])==prov['profiler_executable_sha256'];assert 'Version 2022.4.1.0'in prov['profiler_version']
  old=plan['scratch_cwd'];root=RAW/('batch069'if n=='tsne'else'batch070');scratch=root/n/m/'scratch';fresh=copy.deepcopy(plan)
  for k in ['scratch_cwd','environment_overrides','build_command','profile_target_command','runtime_assets']:fresh[k]=move(plan[k],old,str(scratch))
  copies=[];replaced={};bindings={str(Path(k).resolve()):v for k,v in plan['execution_bindings'].items()};bindings.update({str(Path(k).resolve()):v for k,v in prov.get('additional_current_profile_input_bindings',{}).items()})
  if n=='tsne':
   assert old==tproof['old_scratch']and str(scratch)==tproof['future_scratch']
   for q in tproof['sources']:
    original=q['retained_capture_source'];assert fresh['execution_bindings'].pop(original)==q['retained_capture_sha256'];assert bindings.pop(str(Path(original).resolve()))==q['retained_capture_sha256'];future=q['future_source'];fresh['execution_bindings'][future]=q['future_sha256'];copies.append({'original':original,'original_sha256':q['retained_capture_sha256'],'frozen_copy':q['fresh_source'],'future_copy':future,'future_sha256':q['future_sha256'],'retained_exact_recovered':q['recovered_exact_bytes'],'proof':info(O/'tsne_exact_source_recovery_proof.json')});replaced[original]=q['retained_capture_sha256'];bindings[q['fresh_source']['path']]=q['fresh_source']['sha256'];bindings[q['recovered_exact_bytes']['path']]=q['recovered_exact_bytes']['sha256']
  else:
   assert prov['profile']['stop_reason']=='wall_timeout'and prov['operational_deadline']['requested_seconds']==600
  for p,h in bindings.items():assert sha(p)==h,(n,m,p)
  for a in plan['runtime_assets']:
   for f in a['source_files']:assert sha(f['path'])==f['sha256']
  reverse=copy.deepcopy(fresh)
  for q in copies:del reverse['execution_bindings'][q['future_copy']]
  reverse['execution_bindings'].update(replaced)
  for k in ['scratch_cwd','environment_overrides','build_command','profile_target_command','runtime_assets']:reverse[k]=move(reverse[k],str(scratch),old)
  assert reverse==plan
  op=O/'original_plans'/f'{n}__{m}.json';fp=O/'plans'/f'{n}__{m}'/'plan.json';save(op,plan);save(fp,fresh);timeout=1800 if n=='tsne'else3600
  extras=[];selector=None
  if prov.get('selection_override'):
   se=prov['selection_override'];assert sha(se['path'])==se['sha256'];sel=load(se['path']);dest=O/'selectors'/f'{n}__{m}.json';dest.parent.mkdir(exist_ok=True);dest.write_bytes(Path(se['path']).read_bytes());selectors.append(sel);selector=info(dest);bindings[str(dest)]=sha(dest)
  else:assert prov['command'][3:10]==['--filter-mode','per-launch-config','--kernel-name-base','mangled','--kernel-id','::regex:.*:1','--clock-control']
  for pp in [cap/'plan.json.gz',cap/'profile_provenance.json.gz',cap/'manifest.json']:bindings[str(pp)]=sha(pp)
  bindings[prov['profiler_executable']]=prov['profiler_executable_sha256']
  job={'job_id':n+'__'+m,'group':n,'benchmark':n,'model':m,'expected_contract':{k:ct[k]for k in ['configuration_id','raw_config_sha256','result_sha256']},'expected_latest_profile_id':pid,'previous_profile':info(cap/'profile_provenance.json.gz'),'original_plan':info(op),'prepared_plan':info(fp),'plan_root':str(root),'future_plan_path':str(root/n/m/'plan.json'),'scratch':str(scratch),'old_scratch':old,'timeout_seconds':timeout,'method':None,'legacy_profiler_version':prov['profiler_version'],'legacy_profiler_executable':prov['profiler_executable'],'readonly_bindings':bindings,'source_copies':copies,'restored_original_binding_keys':replaced,'issues':[],'selection':selector,'full_inverse_original_plan_equal':True,'original_qualification':ct['qualification']}
  job['argv']=['/usr/bin/python3',str(W/'tools/v100_roofline_collect.py'),'--source-root',str(W),'--plan-root',str(root),'--benchmark',n,'--skip-qualified-selected','--timeout',str(timeout),'--reserve-mib','2048'];jobs.append(job);inverse.append({'job_id':job['job_id'],'five_operational_fields_path_relocated_only':True,'scientific_metadata_and_historical_proof_preserved':True,'all_original_binding_keys_restored_for_inverse':replaced,'whole_original_plan_inverse_equal':True})
 agg=copy.deepcopy(selectors[0]);agg['selection']={};agg['parent_failures']=[]
 for s in selectors:
  for k in ['benchmark','configuration_id','fixture_description','expected_matching_launches','coverage']:assert s[k]==agg[k]
  agg['selection'].update(s['selection']);agg['parent_failures']+=s['parent_failures']
 assert set(agg['selection'])=={'cuda','cuda_jl','ka','jacc'};ap=O/'selectors/spsort_all4_exact.json';save(ap,agg)
 for j in jobs:
  if j['group']=='spsort':assert agg['selection'][j['model']]==load(j['selection']['path'])['selection'][j['model']];j['selection']=info(ap);j['readonly_bindings'][str(ap)]=sha(ap);j['argv']+=['--selection-overrides',str(ap)]
 tools={str(W/p):sha(W/p)for p in ['tools/v100_roofline_collect.py','tools/v100_roofline_classify.py','tools/v100_profile_write_guard.py','tools/v100_profiler_method.py']};assert tools[str(W/'tools/v100_roofline_collect.py')]=='c21b1edd4c6e0924250b306e1b97ff7ff1371b8407eb92899e089dbdca364116';save(O/'job_inventory.json',{'jobs':jobs,'tool_bindings':tools,'original_full_plan_inverse_proofs':inverse,'all4_SPSORT_selector_union_exact':True,'no_method_or_numeric_workload_change':True,'no_raw_roots_created':not(RAW/'batch069').exists()and not(RAW/'batch070').exists(),'future_binary_scratch_payloads_retained':False,'canonical_C21_full_size_SHA_manifest_then_NEWscratch_cleanup':True})
print('5complete inverseplans currentcontracts/source/methodpins pass; TSNE069freshsources and SPSORT070all4selectors prepared; no rawroots')
