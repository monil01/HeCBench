#!/usr/bin/env python3
"""Restricted CPU-only allreduce method admission. Private proposal, not executed.
Never authorizes GPU execution, writes metadata, invokes processes or changes math.
"""
import base64, hashlib, json, types
from pathlib import Path
METHOD='allreduce_two_device_rank1_private_add_NCU2022_4_1_v1'
MODELS=('cuda','cuda_jl','ka','jacc')
HOLD='profile_completed_allreduce_method_validation_hold'
HELPER_SOURCE=Path(__file__).read_bytes()

def sha(data):return hashlib.sha256(data).hexdigest()
def need(ok,message):
    if not ok:raise ValueError(message)
def load_source(raw,name):
    module=types.ModuleType(name);module.__file__=name
    exec(compile(raw,name,'exec'),module.__dict__)
    return module

def read_raw_slot(slot):
    """Only small adaptation files. Parent payloads are checked separately in full."""
    names=('plan.json','profile_provenance.json','collector_source.py','parser_source.py',
           'write_guard_source.py','counters.csv','derived.json','execution.log',
           'parent_profile_provenance.json','adapter_source.py',
           'allreduce_two_device_review.json','allreduce_cpu_review.json',
           'allreduce_admission_source.py','allreduce_execution_authorization.json')
    return {name:(slot/name).read_bytes()for name in names if(slot/name).is_file()}

def validate(files,root):
    """Raise on missing raw evidence, any mismatch or unreviewed method.
    Root-selected authority is canonical, never a self-declared envelope flag.
    Each call rereads/full-hashes the actual parent and all five complete artifacts;
    validation is not cached and never substitutes hashes for fresh numeric checks.
    """
    plan=json.loads(files['plan.json']);prov=json.loads(files['profile_provenance.json'])
    need(plan['benchmark']=='allreduce'and plan['model']in MODELS,'Restricted allreduce/model authority')
    method=prov['allreduce_two_device_method'];need(method['method_id']==METHOD==plan['profile_method'],'Exact method ID')
    authority=Path(root)/'ontology/studies/v100_roofline_537/completion_reviews/allreduce_two_device/root_cpu_review.json'
    authority_raw=authority.read_bytes();review=json.loads(authority_raw)
    need(authority_raw==files['allreduce_cpu_review.json'],'Canonical root CPU review differs from retained review')
    need(review['method_id']==METHOD and review['root_reviewed_for_cpu_adaptation']is True,'Root method review is not approved')
    need(review['models']==list(MODELS),'Exact four-model method review')
    need(review['canonical_cpu_review_path']==str(authority),'Canonical root authority path')
    need(sha(authority_raw)==method['review_sha256']==plan['profile_method_review']['sha256'],'Root review SHA join')
    need(sha(HELPER_SOURCE)==review['admission_helper']['sha256']==method['admission_helper_sha256'],'Root-approved helper SHA')
    need(files['allreduce_admission_source.py']==HELPER_SOURCE,'Retained helper source differs from active reviewed helper')
    need(sha(files['adapter_source.py'])==review['adapter_sha256']==method['adapter_sha256'],'Root-approved frozen adapter SHA')
    need(sha(files['collector_source.py'])==review['runner']['sha256']==prov['collector_sha256'],'Actual custom collector identity')
    need(sha(files['parser_source.py'])==review['parser_source']['sha256'],'Exact reviewed parser identity')
    envelope_raw=files['allreduce_two_device_review.json'];envelope=json.loads(envelope_raw)
    need(sha(envelope_raw)==method['archive_sha256']and envelope['method_id']==METHOD and envelope['review']==review,'Envelope authority join')
    parent_raw=files['parent_profile_provenance.json'];parent=json.loads(parent_raw)
    need(sha(parent_raw)==method['parent_provenance_sha256']==envelope['parent_provenance_sha256'],'Actual parent provenance join')
    need(sha(files['plan.json'])==prov['plan_sha256'],'Adapted plan SHA')
    slot=Path(plan['profile_capture_parent']);expected=Path(review['capture_root'])/'allreduce'/plan['model']
    need(slot.is_absolute()and slot.resolve()==expected.resolve()and not slot.is_symlink(),'Exact prospectively reviewed parent tree')
    need((slot/'profile_provenance.json').read_bytes()==parent_raw,'Actual parent provenance no longer available/unchanged')
    original_plan_raw=(slot/'plan.json').read_bytes();original_plan=json.loads(original_plan_raw)
    need(sha(original_plan_raw)==method['parent_plan_sha256']==envelope['parent_plan_sha256'],'Actual original plan SHA')
    need(original_plan==envelope['parent_plan'],'Original plan envelope mismatch')
    for key in ('benchmark','model','configuration_id','reference_result_sha256','raw_config_sha256'):
        if key=='benchmark':need(plan[key]=='allreduce','Benchmark identity')
        else:need(plan[key]==original_plan[key],'Original/adapted contract identity: '+key)
    need(prov['profile']==parent['profile']and prov['builds']==parent['builds'],'CPU adaptation changed process facts')
    need(prov['CPU_capture_adaptation']is True and prov['new_gpu_processes']==prov['new_build_processes']==0,'CPU adaptation process scope')
    # Loading is allowed only after canonical-root authority and exact frozen code
    # checks. Never import arbitrary embedded sources just because they carry SHA.
    adapter=load_source(files['adapter_source.py'],'root_reviewed_allreduce_adapter')
    parser=load_source(files['parser_source.py'],'root_reviewed_allreduce_parser')
    bundle_path=adapter.bound(review['proposal']);bundle=json.loads(bundle_path.read_bytes())
    need(bundle['runner']['sha256']==review['runner']['sha256'],'Runner/proposal source join')
    need(parent['proposal_sha256']==review['proposal']['sha256'],'Captured proposal identity')
    # Old2022 calibration is independently bound; no fallback to the queued2025
    # validation or a differently calibrated architecture/operation convention.
    calibration_path=adapter.bound(review['calibration_review']);calibration=json.loads(calibration_path.read_bytes())
    need(calibration['device_uuid']=='GPU-c122ecbd-d9f3-18a5-42d4-1535af37e56d'and calibration['architecture']=='sm_70'and '2022.4.1'in calibration['tool_versions']['ncu'],'Exact old2022GPU6 calibration')
    for key in ('parser_source','calibration_review','collector_reference_source','ncu_executable'):
        item=review[key];need(bundle['all_readonly_bindings'].get(item['path'])==item['sha256'],'Calibration/parser/tool not bound prospectively')
    for item in calibration['files']:
        path=Path(review['source_root'])/item['path']
        need(bundle['all_readonly_bindings'].get(str(path))==item['sha256']and adapter.sha(path)==item['sha256'],'Calibration supporting bytes unavailable or changed')
    operation=prov['operation_calibration']
    need(operation['path']==str(calibration_path)and operation['sha256']==review['calibration_review']['sha256']and operation['half_lane_factor']==2 and operation['default_tensor_conversion_enabled']is False,'Physical half convention changed')
    authorization_path=Path(review['user_authorization_path']);authorization_raw=authorization_path.read_bytes()
    need(sha(authorization_raw)==parent['authorization_receipt_sha256']and authorization_raw==files['allreduce_execution_authorization.json']and parent['authorization']==json.loads(authorization_raw),'Explicit user authorization exact raw receipt join')
    need(parent['authorization']['cpu_capture_adapter_review_sha256']==sha(authority_raw),'User authorization not bound to prospective root CPU review')
    # Preserve exact original tree names, not only the parser's selected payload.
    inventory=envelope['raw_files'];prefix='allreduce/'+plan['model']+'/'
    expected_inventory=[]
    for row in inventory:
        need(row['path'].startswith(prefix),'Cross-slot raw inventory entry')
        expected_inventory.append({**row,'path':row['path'][len(prefix):]})
    need(adapter.archive_tree(slot)==expected_inventory,'Complete parent raw inventory missing/changed; no hash-only fresh verification')
    archive={}
    for item in envelope['compact_exact_raw_files']:
        raw=base64.b64decode(item['base64_bytes'],validate=True)
        need(item['path']not in archive and len(raw)==item['bytes']and sha(raw)==item['sha256'],'Lossless raw envelope item SHA/size')
        p=slot/item['path'];need(not Path(item['path']).is_absolute()and p.resolve().is_relative_to(slot.resolve()),'Archive path escapes parent')
        need(p.read_bytes()==raw,'Actual parent small raw file differs');archive[item['path']]=raw
    need(archive['profile_provenance.json']==parent_raw and archive['plan.json']==original_plan_raw,'Envelope original process/plan absent')
    need(files['execution.log']==(slot/'profile.log').read_bytes(),'Actual execution log differs')
    need(files['counters.csv']==(slot/'counters.csv').read_bytes(),'Actual counters differ')
    # Run original strict CPU validator on the ACTUAL immutable parent tree. This
    # reads every value of all five captures and the actual dual-reference files.
    checks,groups,digest=adapter.validate_slot(slot,original_plan,parent,bundle,review,parser)
    need(method['runtime_checks']==checks,'Frozen method checks differ from freshly recomputed full checks')
    need(adapter.archive_tree(slot)==expected_inventory,'Actual parent changed during full verification')
    need(parent['counter_identity']['counter_sha256']==digest and parent['counter_identity']['parser_sha256']==review['parser_source']['sha256']and parent['counter_identity']['groups']==groups,'Original counter/parse receipt differs from actual complete CSV')
    guard=prov['filesystem_write_guard'];need(guard['tool']==bundle['write_guard']['path']and guard['sha256']==bundle['write_guard']['sha256'],'Actual Landlock tool binding')
    need(sha(files['write_guard_source.py'])==guard['sha256'],'Retained Landlock source identity')
    need(guard['write_directories']==[str(slot.resolve())],'Actual sole writable capture root')
    need(guard['policy']=='Custom runner child-only Landlock restricted to original fresh capture slot; no unrestricted fallback.','Guard scope semantics')
    for phase in [*parent['builds'],parent['profile']]:
        command=phase['command'];need(command[1:5]==[guard['tool'],'--allow-dir',str(slot),'--'],'Actual phase not run under exact slot write guard')
    need(prov['status']=='profile_process_completed','Unaccepted custom adaptation cannot carry accepted derivation')
    report=json.loads(files['derived.json']);need(report['input_sha256']==digest and report['parser_sha256']==review['parser_source']['sha256'],'Derivation parent identity')
    need(len(groups)==report['kernel_count']==len(report['records'])==1,'Exact one sampled add')
    need(all(record['scope_context']==adapter.method_context(plan)and record['fp16_lane_factor']==2 and record['fp16_convention_evidence']==operation and record['tensor_convention_enabled']is False for record in report['records']),'Frozen geometric context/convention changed')
    need(prov['profile_device_identity_verified']and prov['expected_device_pci']=={'domain':0,'bus':203,'device':0},'Actual PCI qualification')
    return {'status':'root_audited_rank1_sample','method_id':METHOD,'root_cpu_review_sha256':sha(authority_raw),
            'admission_helper_sha256':sha(HELPER_SOURCE),'parent_provenance_sha256':sha(parent_raw),
            'all5_complete_artifacts_freshly_verified':True,'whole_ROI_claim':False,
            'process_unit_accounting':'CPU adaptation launches0 GPU/build processes; original parent1 monitored profiler/MPI launcher and '+str(len(parent['builds']))+' build launchers retained. Underlying rank/compiler/device-JIT counts unknown. Deduplicate any separately frozen parent by parent provenance SHA.'}
