"""Offline SU3 tmpfs evidence admission; byte inputs only, no live SDK/path reads."""
import hashlib
import json
NAMES=('tmpfs_storage_policy_source.py','tmpfs_storage_review.json','tmpfs_original_plan.json',
       'tmpfs_native_original.cu','tmpfs_native_overlay.cu','tmpfs_evidence_validator.py','tmpfs_storage_evidence.json')
HOLD='tmpfs_storage_evidence_validation_hold'
def sha(data):return hashlib.sha256(data).hexdigest()
def require(ok,message):
    if not ok:raise ValueError('Tmpfs evidence: '+message)
def substitute(value,old,new):
    if isinstance(value,str):return value.replace(old,new)
    if isinstance(value,list):return [substitute(x,old,new)for x in value]
    if isinstance(value,dict):
        result={}
        for k,v in value.items():
            key=substitute(k,old,new);require(key not in result,'normalized key collision');result[key]=substitute(v,old,new)
        return result
    return value
def validate(prov,files):
    # Missing optional tmpfs metadata means ordinary historical path, not approval.
    if not prov.get('reviewed_tmpfs_storage') and not prov.get('tmpfs_storage_evidence'):
        return None
    p=json.loads(files['plan.json']);link=prov['tmpfs_storage_evidence'];inventory=json.loads(files['tmpfs_storage_evidence.json'])
    require(sha(files['tmpfs_storage_evidence.json'])==link['manifest_sha256'],'snapshot manifest SHA')
    require(sha(files['tmpfs_evidence_validator.py'])==link['validator_sha256'],'validator SHA')
    for name,row in inventory['snapshots'].items():
        require(name in NAMES and name in files,'missing/unknown snapshot '+name)
        require(sha(files[name])==row['sha256'] and len(files[name])==row['bytes'],'snapshot SHA/size '+name)
    if prov['status']=='not_started_tmpfs_storage_preflight_hold':
        return {'status':'retained_preflight_hold','positions_publishable':False,'source':'offline frozen available bytes only; missing inputs remain explicit'}
    require(set(inventory['snapshots'])>={'tmpfs_storage_review.json','tmpfs_storage_policy_source.py','tmpfs_original_plan.json','tmpfs_evidence_validator.py'},'incomplete required snapshots')
    r=json.loads(files['tmpfs_storage_review.json']);s=prov['reviewed_tmpfs_storage'];row=r['models'][p['benchmark']+'/'+p['model']]
    require(p['benchmark']=='su3' and p['model']in ('cuda','cuda_jl','ka','jacc'),'scope')
    require(r['root_reviewed']is True,'root approval absent')
    require(sha(files['tmpfs_storage_review.json'])==s['review_sha256'],'review SHA')
    require(sha(files['tmpfs_storage_policy_source.py'])==r['helper_sha256'],'policy source SHA')
    require(r['adopted_collector_sha256']==prov['collector_sha256']==sha(files['collector_source.py']),'collector SHA')
    require(r['guard_sha256']==sha(files['write_guard_source.py']),'guard SHA')
    require(r['evidence_validator_sha256']==link['validator_sha256'],'root validator approval')
    require(r['gpu_uuid']==prov['gpu_uuid']=='GPU-c122ecbd-d9f3-18a5-42d4-1535af37e56d','device UUID')
    require(sha(files['plan.json'])==row['fresh_plan_sha256']==prov['plan_sha256'],'fresh plan SHA')
    require(sha(files['tmpfs_original_plan.json'])==row['original_plan_sha256'],'original plan SHA')
    require(inventory['snapshots']['tmpfs_original_plan.json']['path']==row['original_plan_path'],'original plan path')
    require(inventory['snapshots']['tmpfs_storage_review.json']['path']==s['review_path'],'review path')
    require(prov.get('profiling_auxiliary_retention')=='kept' or prov['status'].startswith('not_started_'),'full auxiliary retention')
    require(row['read_only_bindings'].get(p['reference_result'])==p['reference_result_sha256'] and row['read_only_bindings'].get(row['raw_config_path'])==p['raw_config_sha256'],'selected result/config review bindings')
    old=json.loads(files['tmpfs_original_plan.json']);normalized=substitute(p,p['scratch_cwd'],old['scratch_cwd'])
    require(s['scratch']==p['scratch_cwd'] and s['owned_root']==row['owned_root'],'retained storage paths')
    require(s['limits']==r['limits'],'retained limits')
    for key in ('configuration_id','raw_config_sha256','reference_result_sha256'):
        require(p[key]==row[key]==old[key],'selected contract '+key)
    require(row['path_roles_independently_reviewed']is True and row['storage_only_no_algorithm_ROI_or_checker_change']is True,'scope approval')
    for path,digest in p['execution_bindings'].items():
        require(row['read_only_bindings'].get(path)==digest,'prospective execution binding '+path)
    proof=row.get('native_failure_diagnostic_overlay')
    if p['model']=='cuda':
        require(isinstance(proof,dict)and proof.get('explicit_root_approval')is True,'native overlay approval')
        require(proof.get('matched_occurrences')==1 and proof.get('only_failure_diagnostic_literal_changed')is True and proof.get('kernel_math_timer_checker_bytes_unchanged')is True,'native correction scope')
        require(inventory['snapshots']['tmpfs_native_original.cu']['path']==proof['original_source_archive'] and inventory['snapshots']['tmpfs_native_overlay.cu']['path']==proof['fresh_source_path'],'native snapshot paths')
        original=files['tmpfs_native_original.cu'];fresh=files['tmpfs_native_overlay.cu'];op=proof['original_source_path'];np=proof['fresh_source_path']
        require(sha(original)==proof['original_source_sha256']==old['execution_bindings'][op],'original native source SHA')
        require(sha(fresh)==proof['fresh_source_sha256']==p['execution_bindings'][np],'fresh native source SHA')
        require(proof['old_scratch_prefix']==old['scratch_cwd']and proof['new_scratch_prefix']==p['scratch_cwd'],'native prefix join')
        before=(old['scratch_cwd']+'/profile_outputs/private_capture/native_warm_').encode();after=(p['scratch_cwd']+'/profile_outputs/private_capture/native_warm_').encode()
        require(original.count(before)==fresh.count(after)==1 and original.count(old['scratch_cwd'].encode())==fresh.count(p['scratch_cwd'].encode())==1,'native unique literal')
        require(original.replace(before,after,1)==fresh and fresh.replace(after,before,1)==original,'native exact byte inverse')
        require(normalized['execution_bindings'][op]==sha(fresh),'sole native binding change')
        normalized['execution_bindings'][op]=sha(original)
    else:require(proof is None,'native exception on Julia')
    require(normalized==old,'full original/fresh plan inverse')
    return {'status':'validated','source':'offline frozen bytes only','positions_publishable':prov['status']=='profile_process_completed','review_sha256':s['review_sha256'],'validator_sha256':link['validator_sha256'],'scope':'Storage evidence only; profiler method, scientific checks, postgate and numerical applicability remain separate.'}
