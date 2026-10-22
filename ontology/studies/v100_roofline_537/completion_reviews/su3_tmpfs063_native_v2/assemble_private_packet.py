# Own CPU text/JSON authoring only; does not import proposed source.
from pathlib import Path
import hashlib,json,difflib,datetime
D=Path(__file__).resolve().parent;V1=Path('/tmp/roofline-completion-20261021/su3_tmpfs063_plan_proposal');R=Path('/home/imo/HeCBench_julia')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(4<<20),b''):h.update(b)
 return h.hexdigest()
def save(name,v):(D/name).write_text(json.dumps(v,indent=2)+'\n')
proof=json.loads((D/'native_overlay_exact_proof.json').read_text())
oldrow=json.loads((V1/'root_review_DRAFT_NOT_APPROVED.json').read_text())
review=json.loads(json.dumps(oldrow));review['proposal_version']=2
native=review['held_models'].pop('su3/cuda');native.pop('hold_reason')
native['fresh_plan_sha256']=sha(D/'collector_plans/su3/cuda/plan.json')
native['path_roles_independently_reviewed']=True
native['native_failure_diagnostic_overlay']=proof
native['read_only_bindings'][proof['fresh_source_path']]=proof['fresh_source_sha256']
native['read_only_bindings'][proof['original_source_archive']]=proof['original_source_sha256']
native['read_only_bindings'][proof['fresh_source_archive']]=proof['fresh_source_sha256']
review['models']['su3/cuda']=native
review['follower_allowed_models']=['cuda','cuda_jl','ka','jacc']
review['helper_sha256']=sha(D/'future_tools/v100_tmpfs_scratch.py')
for model,row in review['models'].items():
 row['read_only_bindings'][str(R/'tools/v100_tmpfs_scratch.py')]=review['helper_sha256']
review['failure_diagnostic_budget_caveat']='Original256MiB auxiliary budget unchanged. Largest failure-only diagnostic335544576B can be partial if resource watchdog stops it; retain every byte/size/SHA and original warmcheck failure. No guaranteed full failure diagnostic or acceptance claimed. Full5success payloads still mandatory.'
save('root_review_DRAFT_NOT_APPROVED.json',review)
s=(V1/'wait_then_collect.py').read_text()
s=s.replace('One collector call, Julia3 only; native remains explicitly held.','One collector call, all4 only after exact native source-overlay approval.')
s=s.replace("MODELS=('cuda_jl','ka','jacc')","MODELS=('cuda','cuda_jl','ka','jacc')")
s=s.replace("for m in ('cuda',*MODELS)","for m in MODELS")
s=s.replace("'native_held':True","'native_held':False")
s=s.replace("'Only Julia3 approved; native held'","'All4 exact reviewed models required'")
s=s.replace(" need(storage['execution_permission']is True,'Explicit reviewed tmpfs execution permission absent')"," need(storage['execution_permission']is True,'Explicit reviewed tmpfs execution permission absent')\n need(storage['models']['su3/cuda']['native_failure_diagnostic_overlay']['explicit_root_approval']is True,'Explicit native source overlay approval required before staging')")
s=s.replace("   # No payload copies: eligible Julia targets retain canonical read-only sources/assets.","   # No payload copies. Julia sources remain canonical; native exact4507B host-path copy only.")
s=s.replace("    for sub in ('profile_outputs','ncu_tmp'):mkdir_private(scratch/sub)\n    out=", """    for sub in ('profile_outputs','ncu_tmp'):mkdir_private(scratch/sub)
    copied=0
    if model=='cuda':
     overlay=storage['models']['su3/cuda']['native_failure_diagnostic_overlay']
     need(overlay['explicit_root_approval']is True,'Separate native path-only approval required')
     mkdir_private(scratch/'profile_source');mkdir_private(scratch/'profile_outputs/private_capture')
     archived=Path(overlay['fresh_source_archive'])
     need(sha(archived)==overlay['fresh_source_sha256'],'Native approved copy bytes changed')
     target=scratch/'profile_source/study_native.cu'
     need(str(target)==overlay['fresh_source_path'],'Exact native source destination required')
     with target.open('xb')as f:f.write(archived.read_bytes());f.flush();os.fsync(f.fileno())
     need(sha(target)==overlay['fresh_source_sha256'],'Staged native copy differs');copied=target.stat().st_size
    out=""")
s=s.replace("'source_copy_bytes':0","'source_copy_bytes':copied")
(D/'wait_then_collect.py').write_text(s)
(D/'follower_minimal.patch').write_text(''.join(difflib.unified_diff((V1/'wait_then_collect.py').read_text().splitlines(True),s.splitlines(True),fromfile='v1/wait_then_collect.py',tofile='v2/wait_then_collect.py')))
launch=json.loads((V1/'launch_review_DRAFT_NOT_APPROVED.json').read_text())
launch['proposal_version']=2;launch['controller_sha256']=sha(D/'wait_then_collect.py')
launch['eligible_models']=['cuda','cuda_jl','ka','jacc'];launch['native_held']=False
launch['future_tool_bindings'][str(R/'tools/v100_tmpfs_scratch.py')]=review['helper_sha256']
launch['storage_review_required_plan_sha256']={k:v['fresh_plan_sha256']for k,v in review['models'].items()}
launch['root_review_requirements']=[
 'New explicit root approval for this exact native failure-prefix/hash pair only; draft nested explicit_root_approval remains false.',
 'Collector0226 stays byte-identical additional filename; canonicalC21 remains unchanged for pending059060/062.',
 'Only helper/a4 follower source changes described in exact diffs; new frontier must pin helperV2.',
 'All4 fivefullsuccess payloads/checks identical; original256MiBaux budget unchanged, failure diagnostic possiblypartial fullyretained.',
 'No execution/adoption/tmpfs writes by author; both fresh root-approved review copies required.']
# Written after report/README at finalization to bind all immutable private inputs.
launch['packet_bindings']={str(p):sha(p)for p in sorted(D.rglob('*'))if p.is_file()and p.name not in ('launch_review_DRAFT_NOT_APPROVED.json','DELIVERY.json','usage_receipt.json','controller_receipt.json')}
save('launch_review_DRAFT_NOT_APPROVED.json',launch)
rows=json.loads((V1/'all_four_plan_join_proofs.json').read_text())
for row in rows:
 row['private_symbolic_plan']=str(D/'collector_plans/su3'/row['model']/'plan.json');row['fresh_plan_sha256']=sha(row['private_symbolic_plan']);row['execution_eligible_after_root_review']=True
 if row['model']=='cuda':
  row['inverse_exact_entire_JSON_including_keys']=False
  row['inverse_exact_after_ONE_native_execution_binding_SHA_normalization']=True
  row['native_overlay_exact_proof']=proof
save('all_four_plan_join_proofs.json',rows)
save('scratch_copy_inventory.json',{'all4_runtime_fixture_copy_bytes':0,'canonical_fixture_readonly_bytes':673919456,
 'files':[{'model':'cuda','source':proof['fresh_source_archive'],'source_sha256':proof['fresh_source_sha256'],
           'future_destination':proof['fresh_source_path'],'bytes':proof['fresh_bytes'],'role':'exact host warmup-failure diagnostic path-only copiedsource'}],
 'existing_readonly_compiler_helper_inventory_from_v1':{'path':str(V1/'scratch_copy_inventory.json'),'sha256':sha(V1/'scratch_copy_inventory.json')},
 'future_native_directories':['scratch/profile_source','scratch/profile_outputs/private_capture'],
 'future_all4_five_payload_bytes':26956993280,'max_owned_root_logical_cap_bytes':32*1024**3,
 'external_writes_by_author':0,'full_inputs_not_copied':True,'no_deletion':True})
save('minimality_and_history_proof.json',{'v1_delivery':{'path':str(V1/'DELIVERY.json'),'sha256':sha(V1/'DELIVERY.json')},
 'collector_bytes_unchanged':sha(D/'future_tools/v100_roofline_collect_tmpfs.py')==sha(V1/'future_tools/v100_roofline_collect_tmpfs.py'),
 'Julia3_plan_bytes_unchanged':{m:sha(D/'collector_plans/su3'/m/'plan.json')==sha(V1/'collector_plans/su3'/m/'plan.json')for m in ('cuda_jl','ka','jacc')},
 'native_only_plan_change':'Exactly one /execution_bindings/<fresh_native_source> value old4bcc→new962d, beyond the already approved inverse scratch-prefix change.',
 'historical_retained_proof':'remaining_completion_proof copy_sha256 intentionally remains original4bcc; new execution-binding hash and explicitly approved exact overlay record are current authority. No broad metadata/hash rewriting.',
 'no_new_math_or_source_compiler_feedback_attempts':True,'scope':'Shared profiling storage/diagnostic path adaptation, not translation.'})
print(json.dumps({'helper_v2_sha256':review['helper_sha256'],'follower_v2_sha256':launch['controller_sha256'],'native_fresh_plan_sha256':native['fresh_plan_sha256']}))
