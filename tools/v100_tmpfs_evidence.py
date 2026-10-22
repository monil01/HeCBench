"""Offline SU3 tmpfs evidence admission; byte inputs only, no live SDK/path reads."""
import hashlib
import json
import math
import posixpath
import re
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

PAYLOAD_BYTES=1347849664
PAYLOAD_COUNT=336962416
TRIAL_REGION='sum_seven_original_AoS_complex_matrix_batches_three_calls_completion'
SITES=(1,16,17,33,257,4096,1048576)

def clean_path(value):
    require(isinstance(value,str)and value!=''and '\x00'not in value,'invalid retained path')
    require('..'not in value.split('/'),'parent path escape')
    require(posixpath.normpath(value)==value,'noncanonical retained path')
    return value

def completed_checks(prov,p,s,files):
    """Frozen source-contract/log/manifest joins; NOT raw payload value recertification."""
    process=prov['profile']
    require(type(process['returncode'])is int and process['returncode']==0,'completed process rc')
    require(process['stop_reason']is None,'completed process stop reason')
    require(process['external_gpu_activity']==[],'completed foreign activity')
    require(prov['profiling_auxiliary_retention']=='kept','full fresh raw retention required')
    raw=files['profiling_auxiliary_manifest.json']
    require(sha(raw)==prov['auxiliary_manifest_sha256']==s['manifest_sha256'],'full auxiliary manifest SHA joins')
    manifest=json.loads(raw);require(isinstance(manifest,list),'auxiliary manifest schema')
    table={}
    for entry in manifest:
        name=clean_path(entry['path'])
        require(not name.startswith('/')and name not in table,'absolute/duplicate auxiliary path')
        require(type(entry['bytes'])is int and entry['bytes']>=0,'auxiliary byte count')
        require(isinstance(entry['sha256'],str)and re.fullmatch('[0-9a-f]{64}',entry['sha256'])is not None,'auxiliary SHA encoding')
        table[name]=entry
    scratch=clean_path(p['scratch_cwd']);prefix=''if p['model']=='cuda'else'profile_outputs/'
    names=[prefix+'su3_trial'+str(i)+'.bin'for i in range(1,6)]
    require(all(name in table and table[name]['bytes']==PAYLOAD_BYTES for name in names),'five complete payload manifest entries')
    declared=prov['large_artifact_validation']
    require(declared['all_declared_payload_files_complete']is True,'full-five size validation absent')
    require(len(declared['files'])==5,'full-five size row count')
    sizes={}
    for row in declared['files']:
        name=clean_path(row['path']);require(name not in sizes,'duplicate declared payload')
        require(type(row['actual_bytes'])is int and type(row['expected_bytes'])is int and row['actual_bytes']==row['expected_bytes']==PAYLOAD_BYTES,'declared payload bytes')
        sizes[name]=row
    expected=[scratch+'/'+name for name in names]
    require(set(sizes)==set(expected),'declared payload paths differ')
    policy=prov['large_artifact_policy']
    require(policy['benchmark']=='su3'and policy['model']==p['model'],'large payload policy identity')
    require(policy['payload_bytes']==5*PAYLOAD_BYTES and policy['max_file_bytes']==2*1024**3,'full payload/per-file budget changed')
    require(policy['auxiliary_allowance_bytes']==256*1024**2 and p['runtime_asset_bytes']==0 and p['runtime_assets']==[],'original auxiliary/no-copy policy changed')
    log=files['execution.log'];require(sha(log)==process['log_sha256'],'actual completed execution log SHA')
    lines=log.decode('utf8').splitlines()
    guard=prov['filesystem_write_guard']
    require(guard['sha256']==sha(files['write_guard_source.py']),'retained actual guard identity')
    directories=guard['write_directories']
    require(isinstance(directories,list)and scratch in directories and len(directories)==len(set(directories)),'exact scratch guard leaf required')
    for name in directories:
        name=clean_path(name)
        require(name==scratch or not(scratch.startswith(name.rstrip('/')+'/')),'broad tmpfs ancestor write allowance')
        require(not name.startswith('/dev/shm/')or name==scratch,'other external tmpfs write allowance')
    require(guard['reviewed_tmpfs_scratch']['review_sha256']==s['review_sha256'],'actual guard review join')
    command=process['executed_command'];require(isinstance(command,list)and len(command)>2,'actual guarded process command absent')
    expected_command=[command[0],guard['tool']]
    for name in directories:expected_command+=['--allow-dir',name]
    expected_command+=['--']+prov['command']
    require(command==expected_command,'guarded command differs')
    require(lines and lines[0].startswith('V100_WRITE_GUARD '),'actual guard log attestation absent')
    attestation=json.loads(lines[0][len('V100_WRITE_GUARD '):])
    require(attestation['write_directories']==directories and type(attestation['landlock_abi'])is int and attestation['landlock_abi']>=3 and attestation['privilege']=='existing unprivileged user; no_new_privs','actual guard attestation differs')
    trials=[json.loads(line[len('STUDY_TRIAL '):])for line in lines if line.startswith('STUDY_TRIAL ')]
    require([row['trial']for row in trials]==[1,2,3,4,5],'exact full-five actual trial sequence required')
    for row,relative in zip(trials,names):
        require(row['correct']is True and row['jit_excluded']is True,'actual full scientific/JIT check failed')
        require(row['internal_repeats']==3 and row['timing_region']==TRIAL_REGION,'actual timed contract changed')
        duration=row['time_seconds'];require(type(duration)in(int,float)and math.isfinite(duration)and duration>=0,'nonfinite/negative source timing')
        cases=row['case_times_seconds'];require(len(cases)==7 and all(type(x)in(int,float)and math.isfinite(x)and x>=0 for x in cases),'seven source case timings required')
        artifact=row['output_artifact'];require(artifact['dtype']=='uint32'and artifact['count']==PAYLOAD_COUNT and artifact['shape']==[PAYLOAD_COUNT]and artifact['byte_order']=='little','full source observable schema changed')
        name=clean_path(artifact['path']);absolute=name if name.startswith('/')else scratch+'/'+name
        require(absolute==scratch+'/'+relative,'actual trial artifact path differs')
    if p['model']=='cuda':
        warm=[]
        for line in lines:
            if line.startswith('STUDY_WARM '):
                match=re.fullmatch(r'STUDY_WARM sites=(\d+) correct=(\d+)',line)
                require(match is not None,'malformed native warm scientific marker')
                warm.append((int(match[1]),int(match[2])))
        require(warm==[(n,1)for n in SITES],'seven native warm full-AoS checks required')
    return {'five_actual_correct_trials':True,'full_five_manifest_entries':True,
        'auxiliary_manifest_sha256':sha(raw),'execution_log_sha256':sha(log),
        'profile_rc0_no_stop_or_foreign_activity':True,'actual_guard_attestation_joined':True,
        'payload_value_recertification':False,
        'scope':'Retained source-defined verdicts and complete SHA/size/path inventory, not a new full raw-byte numerical comparison or live tmpfs lookup.'}

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
    completed=None
    if prov['status']=='profile_process_completed':completed=completed_checks(prov,p,s,files)
    return {'status':'validated','source':'offline frozen bytes only','positions_publishable':prov['status']=='profile_process_completed',
        'review_sha256':s['review_sha256'],'validator_sha256':link['validator_sha256'],
        'completed_source_contract_checks':completed,
        'noncompleted_attempt_qualification':None if completed else 'Retained failed/held process evidence only; no positions and no inferred missing checks.',
        'scope':'Storage evidence only; independent profiler method/postgate and original numerical applicability remain separate. No full raw payload value recertification.'}
