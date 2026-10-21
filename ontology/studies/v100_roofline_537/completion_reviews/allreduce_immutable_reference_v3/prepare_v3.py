"""CPU data/source-copy preparation only. No imports/execution of study tools."""
from pathlib import Path
import ast,copy,datetime,difflib,hashlib,json,os
D=Path(__file__).resolve().parent
V2=Path('/tmp/roofline-completion-20261021/allreduce_two_gpu_proposal')
R=Path('/home/imo/HeCBench_julia')
OLD=str(R/'tools/v100_roofline_collect.py')
H='32a9cfeba983dc3ae8e5d9b8bb85ca4e7736821be0a9c206bb57b80c7471da4c'
NEW=str(D/'immutable_reference/v100_roofline_collect_32a9cfeba983.py')
def digest(b):return hashlib.sha256(b).hexdigest()
def rec(p):return {'path':str(p),'sha256':digest(p.read_bytes()),'bytes':p.stat().st_size}
def dump(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
original_raw=(V2/'proposal.json').read_bytes();v2=json.loads(original_raw)
assert digest(original_raw)=='4a9505298d02fe52f742feaeefa8b2daea1793835f3686031d62e2f927e59fe4'
runner_raw=(V2/'run_proposal.py').read_bytes();assert digest(runner_raw)=='b7563cba2dcf1fbd399695b3b96e6676c63e1b8f4ed238eea09c38948417f642'
reference=Path(OLD).read_bytes();assert digest(reference)==H
Path(NEW).parent.mkdir(exist_ok=False)
with Path(NEW).open('xb')as f:f.write(reference)
Path(NEW).chmod(0o444)
# Exact source AST review: only parser/checker use exec; all other calls and argv
# remain untouched. The collector source appears only as a verified/snapshotted
# reference path in the data maps and never as an executable command.
ast.parse(runner_raw.decode())
changes=[]
def replace(v,old,new,path='$'):
 if isinstance(v,dict):
  out={}
  for k,x in v.items():
   nk=k.replace(old,new)
   if nk!=k:changes.append({'path':path,'role':'dictionary_key','before':k,'after':nk})
   out[nk]=replace(x,old,new,path+'.'+nk)
  assert len(out)==len(v)
  return out
 if isinstance(v,list):return [replace(x,old,new,path+'['+str(i)+']')for i,x in enumerate(v)]
 if isinstance(v,str):
  nv=v.replace(old,new)
  if nv!=v:changes.append({'path':path,'role':'string_value','before':v,'after':nv})
  return nv
 return v
v3=replace(v2,OLD,NEW)
# Nonfunctional revision metadata only. All other fields have a one-path inverse.
v3['revision']=3
changes.append({'path':'$.revision','role':'nonfunctional_revision_metadata','before':2,'after':3})
dump(D/'proposal_v3.json',v3)
inverse=replace(v3,NEW,OLD);inverse['revision']=2
assert inverse==v2
model_proof={}
for name,obj in v2['models'].items():
 n=v3['models'][name];back=replace(n,NEW,OLD)
 b1=json.dumps(obj,indent=2).encode();b2=json.dumps(back,indent=2).encode()
 assert b1==b2
 exact_fields=['build_commands','rank_target_argv','rank_dispatch','rank_worker','session_entry','accepted_MPI_launcher','environment_overrides','ncu_executable','ncu_arguments','kernel_regex','required_metric_units','required_numeric_device_attributes','counter_parser','accepted_model_contract','configuration_id','reference_result_sha256','raw_config_sha256']
 assert all(obj[k]==n[k]for k in exact_fields)
 model_proof[name]={'whole_object_equal_after_inverse_onepath_substitution':True,'serialized_bytes_after_inverse_equal':True,'original_model_sha256':digest(b1),'v3_model_sha256':digest(json.dumps(n,indent=2).encode()),'inverse_model_sha256':digest(b2),'only_changed_role':'execution_bindings key for nonexecuted collector reference; digest unchanged','exact_unchanged_fields':exact_fields}
assert len(v3['all_readonly_bindings'])==len(v2['all_readonly_bindings'])==71
assert OLD not in json.dumps(v3)
assert v3['runner']==v2['runner']
assert (V2/'proposal.json').read_bytes()==original_raw and (V2/'run_proposal.py').read_bytes()==runner_raw
# The diff is the exact payload transformation, not command execution.
(D/'proposal_v2_to_v3.diff').write_text(''.join(difflib.unified_diff(original_raw.decode().splitlines(True),(D/'proposal_v3.json').read_text().splitlines(True),fromfile='immutable_v2/proposal.json',tofile='private_v3/proposal_v3.json')))
proof={'schema_version':1,'created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'Source/data proof only, not build/test/GPU/runtime validation or permission.','v2_proposal':rec(V2/'proposal.json'),'v3_proposal':rec(D/'proposal_v3.json'),'runner_same_original_bytes_path':rec(V2/'run_proposal.py'),'live_reference_read_sha256':H,'frozen_reference':rec(Path(NEW)),'all_readonly_binding_count_before_after':[71,71],'whole_proposal_equal_after_inverse_path_and_revision_metadata':True,'model_proofs':model_proof,'functional_changes':[x for x in changes[:8]if x.get('role')!='nonfunctional_revision_metadata'],'nonfunctional_revision_metadata':{'old':2,'new':3},'permission_objects_unchanged':all(v2[k]==v3[k]for k in ['permission','runtime_gates','limits','physical_GPU_UUIDs','queue_policy','device_attestation_contract']),'all_targets_still_verified_against_same_v2_bindings':[],'original_v2_unchanged':True,'GPU_build_import_test_staging_runs':0}
# Read/hash every original bound file once; no source or runtime method is loaded.
for path,h in v2['all_readonly_bindings'].items():
 assert digest(Path(path).read_bytes())==h
 proof['all_targets_still_verified_against_same_v2_bindings'].append({'path':path,'sha256':h,'byte_role':'reference-only relocated'if path==OLD else'unchanged'})
dump(D/'source_equivalence_review.json',proof)
t=json.loads(Path('/tmp/roofline-completion-20261021/allreduce_integration_patch/root_cpu_review.template.json').read_text())
t['proposal']=rec(D/'proposal_v3.json');t['collector_reference_source']=rec(Path(NEW));t['root_reviewed_for_cpu_adaptation']=False
t['status']='TEMPLATE_ONLY_V3_REFERENCE_DECOUPLED_NOT_CANONICAL_REVIEW_NOT_AUTHORIZED';dump(D/'prospective_root_cpu_review.template.json',t)
# root controls output and permission: this is an argv TEMPLATE, not a launcher.
dump(D/'root_launch_template.json',{'runner':rec(V2/'run_proposal.py'),'argv':['/usr/bin/python3',str(V2/'run_proposal.py'),'--proposal',str(D/'proposal_v3.json'),'--output-root','ROOT_REVIEWED_FRESH_NONBATCH_PATH','--execute','--authorization','ROOT_ACTUAL_EXPLICIT_GPU5_AUTHORIZATION_PATH','--authorization-sha256','ROOT_ACTUAL_AUTHORIZATION_SHA','--queue-wait-seconds','86400'],'execution_authorized':False,'must_not_execute_template':True,'prospective_authorization_requirement':'New proposalSHA plus unchanged runnerSHA and actual root CPU-reviewSHA; explicit userGPU5 permission still required.'})
print(json.dumps({'v3':rec(D/'proposal_v3.json'),'runner':rec(V2/'run_proposal.py'),'reference':rec(Path(NEW)),'proof':rec(D/'source_equivalence_review.json'),'models_inverse_equal':True,'GPU_build_staging_runs':0},indent=2))
