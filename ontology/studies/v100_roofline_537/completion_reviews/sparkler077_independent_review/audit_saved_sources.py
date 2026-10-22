from pathlib import Path
import json,hashlib,fcntl
D=Path('/tmp/roofline-completion-20261021/sparkler077_private_UCX_proposal');O=Path(__file__).parent;H=lambda b:hashlib.sha256(b).hexdigest();F=lambda p:H(Path(p).read_bytes());load=lambda p:json.loads(Path(p).read_bytes())
raw=(D/'DELIVERY.json').read_bytes();assert H(raw)=='f423c7ac7f8c47cd46905ef855191ff19fbb8435b817fd3b5113760beb89813b';delivery=json.loads(raw)
for n,e in delivery['files'].items():b=(D/n).read_bytes();assert H(b)==e['sha256']and len(b)==e['bytes']
s=load(D/'launch_spec.json');assert len(s['bindings'])==187
with Path('/tmp/hecbench-roofline-metadata.lock').open('r')as lock:
 fcntl.flock(lock,fcntl.LOCK_SH);before={p:F(p)for p in s['bindings']};assert before==s['bindings'];ct=load(s['ledger'])['accepted_named_contract'];assert ct==load(s['accepted_contract_snapshot'])
def rp(x,a,b):
 if isinstance(x,str):return x.replace(a,b)
 if isinstance(x,list):return[rp(v,a,b)for v in x]
 if isinstance(x,dict):return{rp(k,a,b):rp(v,a,b)for k,v in x.items()}
 return x
models=[];aliases=0;sources=0
for job in s['jobs']:
 m=job['model'];old=load(D/'original071'/m/'plan.json');new=load(D/'plans'/m/'plan.json');pr=load(D/'proofs'/m/'plan_inverse.json');assert F(D/'original071'/m/'plan.json')==pr['old_plan_sha256'];assert F(D/'plans'/m/'plan.json')==pr['new_plan_sha256']==job['plan_sha256'];assert F(new['sourcebound_UCX_POSIX_recovery']['parent071_plan']['path'])==pr['old_plan_sha256']
 for k,v in [('configuration_id','configuration_id'),('reference_result_sha256','result_sha256'),('raw_config_sha256','raw_config_sha256')]:assert old[k]==new[k]==ct[v]
 assert set(pr['changed_top_level_fields'])=={k for k in set(old)|set(new)if old.get(k)!=new.get(k)};inv=dict(new)
 for field,change in pr['changes'].items():
  assert old.get(field)==change['before']and new.get(field)==change['after']
  if field not in old:assert field=='sourcebound_UCX_POSIX_recovery';inv.pop(field)
  else:inv[field]=change['before']
 assert inv==old
 for x in pr['source_byte_inverses']:
  b=Path(x['new_source_snapshot']).read_bytes();assert H(b)==x['new_sha256'];oldb=(D/'original071'/m/x['name']).read_bytes();assert H(oldb)==x['original071_sha256']==old['execution_bindings'][x['original071_copy_path']];a=x['fresh_slot_prefix']['before'].encode();c=x['fresh_slot_prefix']['after'].encode();add=x['added_text'].encode()
  if add:assert x['name']=='study_launch.py'and add==b"ucx_posix=mpi_base/'ucx_posix';ucx_posix.mkdir(mode=0o700)\nenv['UCX_POSIX_DIR']=str(ucx_posix)\n"and b.count(add)==1;b=b.replace(add,b'',1)
  assert b.replace(c,a)==oldb;sources+=1
 for x in pr['old_missing_binding_aliases']:
  a=x['immutable_alias'];assert F(a['path'])==a['sha256']==x['expected_sha256'];assert Path(a['path']).stat().st_size==a['bytes'];aliases+=1
 om=load(D/'original071'/m/'fresh_compiler_manifest.json');nm=load(D/'sources'/m/'fresh_compiler_manifest.json');assert rp(nm,job['new077_slot'],job['old071_slot'])==om
 models.append({'model':m,'source_inverses':len(pr['source_byte_inverses']),'old_aliases':len(pr['old_missing_binding_aliases']),'full_plan_inverse':True,'compiler_recipe_prefix_only':True})
assert aliases==14 and len(models)==4 and not Path(s['new_root']).exists()
with Path('/tmp/hecbench-roofline-metadata.lock').open('r')as lock:
 fcntl.flock(lock,fcntl.LOCK_SH);assert {p:F(p)for p in before}==before
r={'status':'SOURCE_REVIEW_PASS_NOT_LAUNCH_OR_RUNTIME_APPROVAL','delivery_sha256':H(raw),'controller_sha256':F(D/'controller077.py'),'delivery_members':len(delivery['files']),'readonly_bindings187_twiceSH':True,'source_inverse_count':sources,'alias_count':aliases,'models':models,'new077_rawroot_absent':True,'071_final_completion_or_success_not_inferred':True,'no_imports_build_tests_runtime_GPU_or_shared_mutations':True,'metadata_audit_initial_error':'An inline firstpass indexed absent additive-field oldkey; corrected .get only for explicit newsourcebound metadata, no target/compiler event'};(O/'static_audit.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
