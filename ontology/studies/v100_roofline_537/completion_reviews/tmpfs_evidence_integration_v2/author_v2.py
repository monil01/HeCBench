# CPU source authoring only. Never imports/executes candidates or touches live tools.
from pathlib import Path
import hashlib,difflib,json
D=Path(__file__).resolve().parent;V1=Path('/tmp/roofline-completion-20261021/tmpfs_evidence_integration')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
for name in ('v100_roofline_collect_tmpfs.py','v100_roofline_summarize.py','v100_roofline_audit.py','v100_tmpfs_evidence.py'):
 (D/'bases'/name).write_bytes((V1/'proposed/tools'/name).read_bytes())
validator=(D/'bases/v100_tmpfs_evidence.py').read_text().replace('import json\n','import json\nimport math\nimport posixpath\nimport re\n')
extra='''
PAYLOAD_BYTES=1347849664
PAYLOAD_COUNT=336962416
TRIAL_REGION='sum_seven_original_AoS_complex_matrix_batches_three_calls_completion'
SITES=(1,16,17,33,257,4096,1048576)

def clean_path(value):
    require(isinstance(value,str)and value!=''and '\\x00'not in value,'invalid retained path')
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
                match=re.fullmatch(r'STUDY_WARM sites=(\\d+) correct=(\\d+)',line)
                require(match is not None,'malformed native warm scientific marker')
                warm.append((int(match[1]),int(match[2])))
        require(warm==[(n,1)for n in SITES],'seven native warm full-AoS checks required')
    return {'five_actual_correct_trials':True,'full_five_manifest_entries':True,
        'auxiliary_manifest_sha256':sha(raw),'execution_log_sha256':sha(log),
        'profile_rc0_no_stop_or_foreign_activity':True,'actual_guard_attestation_joined':True,
        'payload_value_recertification':False,
        'scope':'Retained source-defined verdicts and complete SHA/size/path inventory, not a new full raw-byte numerical comparison or live tmpfs lookup.'}

'''
validator=validator.replace('def validate(prov,files):',extra+'def validate(prov,files):')
old="    return {'status':'validated','source':'offline frozen bytes only','positions_publishable':prov['status']=='profile_process_completed','review_sha256':s['review_sha256'],'validator_sha256':link['validator_sha256'],'scope':'Storage evidence only; profiler method, scientific checks, postgate and numerical applicability remain separate.'}"
new="""    completed=None
    if prov['status']=='profile_process_completed':completed=completed_checks(prov,p,s,files)
    return {'status':'validated','source':'offline frozen bytes only','positions_publishable':prov['status']=='profile_process_completed',
        'review_sha256':s['review_sha256'],'validator_sha256':link['validator_sha256'],
        'completed_source_contract_checks':completed,
        'noncompleted_attempt_qualification':None if completed else 'Retained failed/held process evidence only; no positions and no inferred missing checks.',
        'scope':'Storage evidence only; independent profiler method/postgate and original numerical applicability remain separate. No full raw payload value recertification.'}"""
assert validator.count(old)==1;validator=validator.replace(old,new)
(D/'proposed/tools/v100_tmpfs_evidence.py').write_text(validator)
validator_sha=hashlib.sha256(validator.encode()).hexdigest()
summary=(D/'bases/v100_roofline_summarize.py').read_text()
old="(*storage_gate.NAMES,'plan.json','collector_source.py','write_guard_source.py')"
new="(*storage_gate.NAMES,'plan.json','collector_source.py','write_guard_source.py','profiling_auxiliary_manifest.json','execution.log')"
assert summary.count(old)==1;summary=summary.replace(old,new)
(D/'proposed/tools/v100_roofline_summarize.py').write_text(summary)
audit=(D/'bases/v100_roofline_audit.py').read_text()
needle="                    source=files['tmpfs_evidence_validator.py']\n                    require(sha(source)==prov['tmpfs_storage_evidence']['validator_sha256'],'Tmpfs frozen validator SHA mismatch')"
replacement=f"""                    source=files['tmpfs_evidence_validator.py']
                    # Admit known root-reviewed validator identity BEFORE executing retained bytes.
                    # This constant is source-pinned by the root-adopted audit/frontier, not archive self-attestation.
                    require(sha(source)=={validator_sha!r},'Tmpfs validator is not the exact source-reviewed version')
                    require(sha(source)==prov['tmpfs_storage_evidence']['validator_sha256'],'Tmpfs frozen validator SHA mismatch')"""
assert audit.count(needle)==1;audit=audit.replace(needle,replacement)
needle="                        qualifications.append({'profile_id':identity,'reason':'Retained tmpfs preflight rejection; no GPU/build/profile attempt admitted'})"
replacement=needle+"\n                    elif prov['status']!='profile_process_completed':\n                        qualifications.append({'profile_id':identity,'reason':'Retained tmpfs failed/held process evidence; full-success checks not inferred and no positions admitted'})"
assert audit.count(needle)==1;audit=audit.replace(needle,replacement)
(D/'proposed/tools/v100_roofline_audit.py').write_text(audit)
(D/'proposed/tools/v100_roofline_collect_tmpfs.py').write_bytes((D/'bases/v100_roofline_collect_tmpfs.py').read_bytes())
for name in ('v100_tmpfs_evidence.py','v100_roofline_summarize.py','v100_roofline_audit.py','v100_roofline_collect_tmpfs.py'):
 old=(D/'bases'/name).read_text();new=(D/'proposed/tools'/name).read_text()
 (D/(name+'.patch')).write_text(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='v1/'+name,tofile='v2/'+name)))
print(json.dumps({'validator_sha256':validator_sha,'summary_sha256':sha(D/'proposed/tools/v100_roofline_summarize.py'),'audit_sha256':sha(D/'proposed/tools/v100_roofline_audit.py'),'collector_UNCHANGED_sha256':sha(D/'proposed/tools/v100_roofline_collect_tmpfs.py')}))
