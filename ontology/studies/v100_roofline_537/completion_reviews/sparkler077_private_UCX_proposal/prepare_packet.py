from pathlib import Path
import json,hashlib,fcntl,copy,base64
P=Path(__file__).parent;W=Path('/home/imo/HeCBench_julia');OLD=Path('/tmp/hecbench-roofline-v100-20261020/batch071-sparkler-privateMPI');NEW=Path('/tmp/hecbench-roofline-v100-20261020/batch077-sparkler-privateUCX');D=P.parent/'sparkler071_native_failure_review';H=Path('/opt/nvidia/hpc_sdk/Linux_x86_64/2025/comm_libs/12.9/hpcx/hpcx-2.22.1')
sha=lambda b:hashlib.sha256(b).hexdigest();fh=lambda p:sha(Path(p).read_bytes())
def dump(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2)+'\n')
def remap(x,a,z):
 if isinstance(x,str):return x.replace(a,z)
 if isinstance(x,list):return [remap(v,a,z) for v in x]
 if isinstance(x,dict):return {remap(k,a,z):remap(v,a,z) for k,v in x.items()}
 return x
bind={};jobs=[]
def bindfile(p,expected=None):
 h=fh(p);assert expected is None or h==expected,(str(p),h,expected);bind[str(p)]=h
with Path('/tmp/hecbench-roofline-metadata.lock').open('r') as lock:
 fcntl.flock(lock,fcntl.LOCK_SH);ledger=W/'ontology/studies/v100_roofline_537/benchmarks/sparkler.json';contract=json.loads(ledger.read_bytes())['accepted_named_contract']
 # Full ledger is intentionally not frozen here:071 remains active and may
 # legitimately update selected measurement fields. Accepted contract is exact.
 dump(P/'accepted_contract_snapshot.json',contract)
for model in ['cuda','cuda_jl','ka','jacc']:
 oldslot=OLD/'sparkler'/model;newslot=NEW/'sparkler'/model;oldraw=(oldslot/'plan.json').read_bytes();old=json.loads(oldraw);p=copy.deepcopy(old);ar=P/'original071'/model;ar.mkdir(parents=True,exist_ok=True);(ar/'plan.json').write_bytes(oldraw);bindfile(oldslot/'plan.json');bindfile(oldslot/'fresh_compiler_manifest.json')
 assert all(old[a]==contract[b] for a,b in [('configuration_id','configuration_id'),('reference_result_sha256','result_sha256'),('raw_config_sha256','raw_config_sha256')])
 bindfile(old['reference_result'],old['reference_result_sha256']);bindfile(Path(old['reference_result']).with_name('config.json'),old['raw_config_sha256'])
 local={};sourceproof=[];staged=[]
 for patch in old['sparkler_private_MPI_recovery']['patches']:
  before=patch['after_text'];assert sha(before.encode())==patch['copy_sha256']==old['execution_bindings'][patch['copy']]
  name=Path(patch['copy']).name;(ar/name).write_text(before)
  after=before.replace(str(oldslot),str(newslot));addition=""
  if name=='study_launch.py':
   marker="env.update(OMPI_MCA_orte_tmpdir_base=str(sessions),OMPI_MCA_shmem_mmap_relocate_backing_file='1',OMPI_MCA_shmem_mmap_backing_file_base_dir=str(backing),OMPI_MCA_btl_vader_backing_directory=str(vader))\n"
   addition="ucx_posix=mpi_base/'ucx_posix';ucx_posix.mkdir(mode=0o700)\nenv['UCX_POSIX_DIR']=str(ucx_posix)\n";assert after.count(marker)==1;after=after.replace(marker,marker+addition)
  restored=after.replace(addition,'') if addition else after;restored=restored.replace(str(newslot),str(oldslot));assert restored.encode()==before.encode()
  dest=P/'sources'/model/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(after);runtime=str(newslot/'scratch/profile_source'/name);local[patch['copy']]=(runtime,fh(dest));staged.append({'source':str(dest),'destination':runtime,'sha256':fh(dest)})
  sourceproof.append({'name':name,'original071_copy_path':patch['copy'],'original071_sha256':sha(before.encode()),'new_source_snapshot':str(dest),'new_runtime_path':runtime,'new_sha256':fh(dest),'added_text':addition,'fresh_slot_prefix':{'before':str(oldslot),'after':str(newslot)},'exact_full_byte_inverse_equal':True})
 compilerraw=(oldslot/'fresh_compiler_manifest.json').read_bytes();(ar/'fresh_compiler_manifest.json').write_bytes(compilerraw);recipe=remap(json.loads(compilerraw),str(oldslot),str(newslot));cp=P/'sources'/model/'fresh_compiler_manifest.json';dump(cp,recipe);assert remap(recipe,str(newslot),str(oldslot))==json.loads(compilerraw);local[str(oldslot/'fresh_compiler_manifest.json')]=(str(newslot/'fresh_compiler_manifest.json'),fh(cp));staged.append({'source':str(cp),'destination':str(newslot/'fresh_compiler_manifest.json'),'sha256':fh(cp)})
 for key in ['scratch_cwd','environment_overrides','build_command','profile_target_command','runtime_assets']:p[key]=remap(old[key],str(oldslot),str(newslot))
 b={};aliasproof=[]
 for path,h in old['execution_bindings'].items():
  if path in local:
   name,newhash=local[path];b[name]=newhash
  else:
   assert not path.startswith(str(oldslot)),path;bindfile(path,h);b[path]=h
 # All historical missing-input aliases retained exactly, no dropped hashes.
 for alias in old['sparkler_private_MPI_recovery']['historical_missing_binding_aliases']:
  item=alias['immutable_alias'];bindfile(item['path'],alias['expected_sha256']);assert b[item['path']]==alias['expected_sha256'];aliasproof.append(alias)
 p['execution_bindings']=b
 p['sourcebound_UCX_POSIX_recovery']={'parent071_plan':{'path':str(oldslot/'plan.json'),'sha256':fh(oldslot/'plan.json')},'mechanism_packet':{'path':str(D/'DELIVERY.json'),'sha256':fh(D/'DELIVERY.json')},'only_operational_addition':'UCX_POSIX_DIR=new owned mpi_base/ucx_posix directory; UCC/UCX/collective/transport flags unchanged','source_proofs':sourceproof,'historical_aliases_preserved_count':len(aliasproof),'math_checks_timers_metrics_method_unchanged':True,'runtime_recovery_proven':False,'historical_loaded_DSO_identity_claim':False}
 inv={k:{'before':old.get(k),'after':v} for k,v in p.items() if old.get(k)!=v}
 allowed={'scratch_cwd','environment_overrides','build_command','profile_target_command','runtime_assets','execution_bindings','sourcebound_UCX_POSIX_recovery'};assert set(inv)<=allowed
 restored=copy.deepcopy(p)
 for k,change in inv.items():
  if k in old:restored[k]=change['before']
  else:del restored[k]
 assert restored==old
 target=P/'plans'/model/'plan.json';dump(target,p);dump(P/'proofs'/model/'plan_inverse.json',{'old_plan_sha256':sha(oldraw),'new_plan_sha256':fh(target),'changed_top_level_fields':list(inv),'changes':inv,'exact_full_object_inverse_equal':True,'original071_fulltext_retained':str(ar/'plan.json'),'source_byte_inverses':sourceproof,'old_missing_binding_aliases':aliasproof,'all_original_alias_hashes_retained':True,'compiler_manifest_inverse_full_object_equal':True})
 for src in staged:bindfile(src['source'],src['sha256'])
 bindfile(target);bindfile(P/'proofs'/model/'plan_inverse.json')
 jobs.append({'model':model,'plan_source':str(target),'plan_runtime':str(newslot/'plan.json'),'plan_sha256':fh(target),'staged_files':staged,'old071_slot':str(oldslot),'new077_slot':str(newslot),'configuration_id':p['configuration_id'],'reference_result_sha256':p['reference_result_sha256'],'raw_config_sha256':p['raw_config_sha256'],'historical_alias_count':len(aliasproof)})
method=W/'ontology/studies/v100_roofline_537/profiler_methods/ncu2025_2_0_V100_sm70_raw_v1/review.json';mr=json.loads(method.read_bytes());bindfile(method,'65145043726f06195ef78dc89e72ca996b8fbb9217e4f2b8a3226e24e93b5eca');bindfile(method.parent/mr['bundle_file'],mr['bundle_sha256'])
archive=json.loads((method.parent/mr['bundle_file']).read_bytes());man=json.loads(base64.b64decode(archive['files']['manifest.json']['base64']))
for name,digest in man['bindings'].items():
 if name==mr['profiler']['executable'] or name.startswith(str(Path(mr['profiler']['executable']).parent)+'/'):bindfile(name,digest)
for rel in ['tools/v100_roofline_collect.py','tools/v100_profiler_method.py','tools/v100_profile_write_guard.py','tools/v100_roofline_classify.py']:bindfile(W/rel)
bindfile(W/'tools/v100_roofline_collect.py','c21b1edd4c6e0924250b306e1b97ff7ff1371b8407eb92899e089dbdca364116')
for item in json.loads((D/'installed_UCX_mechanism.json').read_bytes())['files']:bindfile(item['path'],item['sha256'])
for name in ['DELIVERY.json','evidence_audit.json','installed_UCX_mechanism.json','future_launcher_inverse.json']:bindfile(D/name)
bindfile(P/'accepted_contract_snapshot.json');bindfile(P.parent/'sparkler071_recovery/DELIVERY.json')
argv=['/usr/bin/python3',str(W/'tools/v100_roofline_collect.py'),'--source-root',str(W),'--plan-root',str(NEW),'--benchmark','sparkler','--timeout','3600','--reserve-mib','2048','--skip-qualified-selected','--profiler-executable',mr['profiler']['executable'],'--profiler-method-review',str(method),'--profiler-method-review-sha256',fh(method)]
dump(P/'launch_spec.json',{'new_root':str(NEW),'old071_root':str(OLD),'source_root':str(W),'ledger':str(ledger),'accepted_contract_snapshot':str(P/'accepted_contract_snapshot.json'),'jobs':jobs,'bindings':bind,'argv':argv,'environment':{'PATH':'/usr/bin:/bin','LD_LIBRARY_PATH':'','LD_PRELOAD':'','PYTHONPATH':'','PYTHONDONTWRITEBYTECODE':'1'},'max_invocations':1,'process_budget_seconds':3600,'required_free_bytes':6168779460,'storage_basis':'800070340B full5primarypayload +3GiB prospective CAS/guard/build/counter margin +2GiBreserve, planning not exact peak or all-CAS completeness bound','071_group_must_be_finished_and_root_reviewed':True,'retained_aliases':sum(j['historical_alias_count'] for j in jobs),'canonical_C21_only_no_future_guard_assumption':True})
print(json.dumps({'models':len(jobs),'aliases':sum(j['historical_alias_count'] for j in jobs),'bindings':len(bind),'rawroot_exists':NEW.exists()}))
