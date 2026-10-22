from pathlib import Path
import json,gzip,hashlib,copy,fcntl
D=Path('/tmp/roofline-completion-20261021/snicit075_conditional_recovery');O=Path(__file__).parent;W=Path('/home/imo/HeCBench_julia');H=lambda b:hashlib.sha256(b).hexdigest();F=lambda p:H(Path(p).read_bytes());load=lambda p:json.loads(Path(p).read_bytes());checks=[]
def ck(k,v,detail=None):checks.append({'check':k,'passed':bool(v),'detail':detail})
def repl(x,a,b):
 if isinstance(x,str):return x.replace(a,b)
 if isinstance(x,list):return[repl(v,a,b)for v in x]
 if isinstance(x,dict):return{repl(k,a,b):repl(v,a,b)for k,v in x.items()}
 return x
ck('exact_DELIVERY',F(D/'DELIVERY.json')=='6ce8b9738ecd3a771ee2c6a50f6e930d7cd0496c6622d36357777e5246ee6077')
for path,r in load(D/'DELIVERY.json')['files'].items():ck('DELIVERY_static_inventory',F(D/path)==r['sha256']and (D/path).stat().st_size==r['bytes'],path)
p=load(D/'proposal075.json');npins=load(D/'source/numpy_bindings.json');dep=load(D/'numpy_elf_dependency_review.json');ck('four_exact_models',[r['model']for r in p['models']]==['cuda','cuda_jl','ka','jacc']);ck('max4',p['maximum_collector_invocations']==4);ck('NumPy_existing_count',len(npins['files'])==1343 and npins['expected_numpy_version']=='2.5.4');ck('ELF31_no_unresolved',len(dep['ELF_metadata_records'])==31 and dep['unresolved']==[])
for path,sha in npins['files'].items():ck('existing_CPU_dependency_SHA',F(path)==sha,path)
for r in dep['ELF_metadata_records']:
 ck('ELF_exact_SHA',F(r['ELF'])==r['ELF_sha256'],r['ELF'])
 for e in r['needed_resolved']:ck('resolved_ELF_SHA',F(e['path'])==e['sha256'],e['path'])
old=(D/'source/study_bridge_original.cu').read_bytes();new=(D/'source/study_bridge_profile.cu').read_bytes();before=p['source_whitelist']['before'].encode();after=p['source_whitelist']['after'].encode();ck('one_exact_bridge_intervention',old.count(before)==1 and old.replace(before,after)==new and new.replace(after,before)==old);ck('original_canonical_bridge_SHA',F(W/'src/snicit-julia/study_bridge.cu')==H(old));ck('post_ROI_only',new.index(after)>new.index(b'total+=std::chrono::duration') and new.index(after)>new.index(b'launch(3,-1)'));ck('unchanged_audit',F(W/'src/snicit-julia/audit_model.py')=='7d6fa088806b1987920080d4dc092ae2d3459196dea8e0188fb3e13a70ad09ba')
held=(D/'controller075.py').read_text();ck('unconditional_execute_hold',"if not a.execute:"in held and "raise RuntimeError('075 EXECUTION HOLD:"in held and 'subprocess.run('not in held and 'subprocess.Popen('not in held)
with open('/tmp/hecbench-roofline-metadata.lock','r')as l:
 fcntl.flock(l,fcntl.LOCK_SH);ledger=load(W/'ontology/studies/v100_roofline_537/benchmarks/snicit.json')
 for r in p['models']:
  m=r['model'];parent=W/r['parent014_profile'];b=(parent/'plan.json.gz').read_bytes();raw=gzip.decompress(b);o=json.loads(raw);n=load(r['prepared_plan']['path']);prior=o['overlay_provenance']['profile_source_patches'][0];retained=prior['text'].encode();ck('exact_parent014_plan',H(b)==r['parent_plan_compressed_sha256']and H(raw)==r['parent_plan_raw_sha256']and load(D/'original014_plans'/m/'plan.json')==o,m);ck('exact_parent_provenance',H(gzip.decompress((parent/'profile_provenance.json.gz').read_bytes()))==r['parent_provenance_sha256'],m);ck('retained014_wrapper_fullbytes',retained==(D/'source/repaired014_wrapper_exact.py').read_bytes()and H(retained)==prior['copy_sha256']==o['execution_bindings'][prior['copy']],m)
  oldslot=str(Path(o['scratch_cwd']).parent);slot=str(Path(r['future_plan']).parent);expected=repl(o,oldslot,slot);wrapperold=prior['copy'].replace(oldslot,slot);future=str(Path(n['scratch_cwd'])/'profile_source');expected['profile_target_command'][0]='/usr/bin/python3';expected['profile_target_command'][1]=future+'/study_run_model.py';julia='/home/imo/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia'
  if m!='cuda':ck('original_nested_Julia_role',o['profile_target_command'][2]=='julia' and r['nested_executable_change'][0]['index']==2,m);expected['profile_target_command'][2]=julia
  expected['execution_bindings'].pop(wrapperold)
  for c in r['copies']:ck('exact_prepared_source_copy_SHA',F(c['source'])==c['sha256']and c['destination']==future+'/'+Path(c['source']).name,m);expected['execution_bindings'][c['destination']]=c['sha256']
  expected['execution_bindings'].update(npins['files'])
  for path in [julia,str(Path('/usr/bin/python3').resolve()),'/usr/bin/nvcc']:expected['execution_bindings'][path]=p['tools'][path]
  idx=expected['build_command'].index(str(W/'src/snicit-julia/study_bridge.cu'));expected['build_command'][idx]=future+'/study_bridge_profile.cu';expected['build_command'][idx:idx]=['-I',str(W/'src/snicit-julia')]
  ck('independent_complete_plan_transform_equality',expected==n,m);ck('new_plan_exact_SHA',F(r['prepared_plan']['path'])==r['prepared_plan']['sha256'],m);ck('all_future_roots_uncreated',not Path(r['future_root']).exists(),m);ck('same_contract',all(ledger['accepted_named_contract'][k]==v for k,v in r['expected_contract'].items()),m)
  for path,sha in o['execution_bindings'].items():
   if path!=prior['copy']:ck('unchanged_original_execution_binding',F(path)==sha,path)
  a=r['argv'];ck('timeout900_reserve2048_skip_no_retainaux',a[a.index('--timeout')+1]=='900'and a[a.index('--reserve-mib')+1]=='2048'and '--skip-qualified-selected'in a and '--retain-auxiliary'not in a,m);ck('admitted_method_exact_review',a[a.index('--profiler-method-review-sha256')+1]=='65145043726f06195ef78dc89e72ca996b8fbb9217e4f2b8a3226e24e93b5eca',m)
 fcntl.flock(l,fcntl.LOCK_UN)
for path,sha in p['tools'].items():ck('actual_prospective_tool_SHA',F(path)==sha,path)
report={'scope':'Independent saved-source and proposed-role review only; held controller not executed','checks_total':len(checks),'checks_passed':sum(r['passed']for r in checks),'blockers':[r for r in checks if not r['passed']],'checks':checks,'source_proposal_reviewable':all(r['passed']for r in checks),'root_launch_admissible':False,'mandatory_hold':'Unreviewed child ownership monitor authority, explicit controller execution hold; new frozen packet required before any staging or launch','runtime_cause_claim':False,'GPU':0,'builds':0,'tests':0,'target_imports':0,'shared_scientific_mutations':False};q=O/'independent_review.json';q.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'checks':len(checks),'blockers':report['blockers'],'source_reviewable':report['source_proposal_reviewable'],'root_launch_admissible':False,'sha256':F(q)}))
