#!/usr/bin/env python3
"""Audit saved roofline evidence and recompute derivations; never launch a GPU.

This checks retained bytes, not the truth of an unproved ROI or old source identity.
The exact hash-bound historical parser is used, rather than silently substituting
the current implementation. Only this repository's retained parser sources run.
"""
import argparse
import base64
import struct
import collections
import datetime
import fcntl
import gzip
import hashlib
import inspect
from v100_profiler_method import LEGACY_VERSION
import json
from pathlib import Path
import tempfile
import types
from v100_roofline_allreduce_admission import validate as validate_allreduce_method


AUDIT_SOURCE = Path(__file__).read_bytes()

def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source-root', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    root = args.source_root.resolve()
    study = root / 'ontology/studies/v100_roofline_537'
    issues, qualifications, attempts = [], [], []
    parsers = {}
    def require(condition, message):
        if not condition:
            raise ValueError(message)
    def frozen(path):
        require(path.resolve().is_relative_to(study), 'Evidence path escapes study')
        return gzip.decompress(path.read_bytes())
    snapshots = {}
    for path in (study / 'tool_snapshots').glob('*.py'):
        data = path.read_bytes()
        snapshots[sha(data)] = data
    for path in (study / 'tool_snapshots').glob('*.py.gz'):
        data = gzip.decompress(path.read_bytes())
        snapshots[sha(data)] = data
    with tempfile.TemporaryDirectory(prefix='roofline-cpu-audit-', dir='/tmp') as temp:
        for manifest_path in sorted((study / 'profiles').glob('*/*/*/manifest.json')):
            identity = str(manifest_path.parent.relative_to(root))
            row = {'profile_id': identity}
            try:
                manifest = json.loads(manifest_path.read_text())
                files = {}
                for entry in manifest['files']:
                    path = root / entry['path']
                    packed = path.read_bytes()
                    data = frozen(path)
                    require(sha(packed) == entry['compressed_sha256'], 'Compressed SHA mismatch: '+str(path))
                    require(len(packed) == entry['compressed_bytes'], 'Compressed size mismatch')
                    require(sha(data) == entry['uncompressed_sha256'], 'Raw SHA mismatch: '+str(path))
                    require(len(data) == entry['uncompressed_bytes'], 'Raw size mismatch')
                    require(entry['original_name'] not in files, 'Duplicate manifest filename')
                    files[entry['original_name']] = data
                prov = json.loads(files['profile_provenance.json'])
                plan = json.loads(files['plan.json'])
                require(prov['gpu_uuid'] == 'GPU-c122ecbd-d9f3-18a5-42d4-1535af37e56d', 'Unexpected device UUID')
                require(sha(files['plan.json']) == prov['plan_sha256'], 'Plan provenance SHA mismatch')
                collector = files.get('collector_source.py', snapshots.get(prov['collector_sha256']))
                require(collector is not None and sha(collector) == prov['collector_sha256'], 'Exact collector source unavailable')
                for name, key in [('config.json','raw_config_sha256'), ('result.json','reference_result_sha256')]:
                    data = frozen(study / 'selected_contracts' / plan['benchmark'] / (name+'.gz'))
                    require(sha(data) == plan[key], 'Selected contract SHA mismatch: '+name)
                row.update(benchmark=plan['benchmark'], model=plan['model'], status=prov['status'], manifest_files=len(files))
                if prov.get('profile_window'):
                    link=plan['profile_window_review'];raw=files['profile_window_review.json']
                    require(sha(raw)==link['sha256']==prov['profile_window']['review']['sha256'], 'Marker window review SHA mismatch')
                    record=json.loads(raw)['models'][link['entry']]
                    require(record==prov['profile_window']['record'] and plan['benchmark']=='rayleighBenardConvection', 'Marker window record mismatch')
                    require(record['reference_result_sha256']==plan['reference_result_sha256'] and record['raw_config_sha256']==plan['raw_config_sha256'], 'Marker window selected contract mismatch')
                    require(record['profile_from_start'] is False and record['capture_all_launches_in_window'] is False, 'Marker coverage was promoted')
                    require('--profile-from-start' in prov['command'] and prov['command'][prov['command'].index('--profile-from-start')+1]=='off', 'Marker window not disabled from process start')
                    for name,digest in record['execution_bindings'].items():require(plan['execution_bindings'].get(name)==digest, 'Marker input not prospectively bound')
                    if prov['status']=='profile_process_completed':
                        logs=files['execution.log'].decode(errors='replace').splitlines()
                        starts=[json.loads(l[len('RBC_PROFILE_WINDOW_BEGIN '):])for l in logs if l.startswith('RBC_PROFILE_WINDOW_BEGIN ')]
                        ends=[json.loads(l[len('RBC_PROFILE_WINDOW_END '):])for l in logs if l.startswith('RBC_PROFILE_WINDOW_END ')]
                        trials=[json.loads(l[len('STUDY_TRIAL '):])for l in logs if l.startswith('STUDY_TRIAL ')]
                        require(len(starts)==len(ends)==1 and starts[0]['trial']==ends[0]['trial']==2 and starts[0]['zero_based_step']==ends[0]['zero_based_step']==240, 'Actual marker window mismatch')
                        require(starts[0]['nu_samples_before']==1 and ends[0]['nu_samples_after']==2 and ends[0]['authored_dispatch_counts']==[3,6,2,11,5,0,1], 'Actual source window dispatch/Nu mismatch')
                        require([t['trial']for t in trials]==list(range(1,6)) and all(t['correct'] is True for t in trials), 'Marker experiment full five checks absent')
                        checks=prov['profile_window_runtime_checks']
                        require(checks['all_pass'] is True and checks['begin']==starts and checks['end']==ends, 'Saved marker derivation mismatch')
                    row['attempt_kind']='retained_scientific_timestep_partial_specializations_not_complete_ROI'
                if prov.get('isolated_assertion_phases'):
                    link=plan['isolated_assertion_review'];raw=files['isolated_assertion_review.json']
                    require(sha(raw)==link['sha256']==prov['isolated_assertion_phases']['review']['sha256'], 'Isolated assertion review SHA mismatch')
                    review=json.loads(raw)['models'][link['entry']]
                    require(review==prov['isolated_assertion_phases']['record'] and plan['benchmark']=='assert' and plan['model'] in ('cuda_jl','ka','jacc'), 'Isolated assertion method mismatch')
                    require(review['positive_command']==plan['profile_target_command'] and review['reference_result_sha256']==plan['reference_result_sha256'] and review['raw_config_sha256']==plan['raw_config_sha256'], 'Isolated assertion contract mismatch')
                    passfiles={}
                    for item in json.loads(files['isolated_assertion_negative_raw.json'])['files']:
                        data=base64.b64decode(item['base64_bytes'],validate=True)
                        require(len(data)==item['bytes'] and sha(data)==item['sha256'], 'Isolated negative raw SHA mismatch')
                        require(item['path'] not in passfiles, 'Duplicate isolated negative path')
                        passfiles[item['path']]=data
                    if prov['status']=='profile_process_completed':
                        pre=json.loads(files['negative_prechecks.json']);post=json.loads(files['joined_execution_contract.json']);outcome=json.loads(files['positive_profile_outcome.json'])
                        require(pre['complete'] is True and pre['all_pass'] is True and len(pre['processes'])==5, 'Five isolated negatives missing')
                        require(post['complete_execution_only_contract'] is True and post['numerical_agreement_claim'] is False and len(post['rows'])==5, 'Full assertion execution-only contract missing')
                        require(outcome['plan_sha256']==sha(files['plan.json']) and outcome['command']==plan['profile_target_command'] and outcome['returncode']==0 and outcome['stop_reason'] is None and outcome['own_session_quiescent'] is True, 'Isolated positive execution not accepted')
                        require(outcome['execution_log']['sha256']==sha(files['execution.log']), 'Isolated positive log mismatch')
                        for phase in ('pre_profile_checks','post_profile_checks'):
                            require(prov[phase]['returncode']==0 and prov[phase]['stop_reason'] is None, 'Isolated check phase failed')
                        perf=[json.loads(l[len('PERF_TRIAL '):])for l in files['execution.log'].decode(errors='replace').splitlines() if l.startswith('PERF_TRIAL ')]
                        require([r['trial']for r in perf]==list(range(1,6)), 'Direct positive five PERF trials missing')
                        for expected_trial,(negative,positive,joined) in enumerate(zip(pre['processes'],perf,post['rows']),1):
                            require(negative['trial']==expected_trial and negative['returncode']==0 and negative['checker_pass'] is True, 'Isolated negative failed')
                            actual={}
                            for key in ('stdout','stderr','combined_log'):
                                item=negative[key];name='negative_raw/'+Path(item['path']).name;data=passfiles[name]
                                require(sha(data)==item['sha256'] and len(data)==item['bytes'], 'Negative receipt raw join mismatch');actual[key]=data
                            require(actual['combined_log']==actual['stdout']+actual['stderr'], 'Negative combined raw mismatch')
                            require([line.split()for line in actual['stdout'].decode().splitlines()if line.startswith('ASSERT_STATUS ')]==[['ASSERT_STATUS','710']], 'Actual isolated CUDA status mismatch')
                            require(all(joined[k]==v for k,v in positive.items()) and joined['correct'] is True and joined['checked_values']==0 and joined['observed_device_assertion_status']==710 and joined['expected_device_assertion_status']==710, 'Joined assertion verdict differs from actual checks')
                    row['attempt_kind']='five_negative_contexts_plus_direct_positive_profile_and_mandatory_CPU_join'
                if prov.get('replay_mode') == 'application':
                    require(plan['benchmark']=='atomicSystemWide', 'Unreviewed application replay benchmark')
                    review_bytes=files['application_replay_review.json']
                    link=plan['application_replay_review']
                    require(sha(review_bytes)==link['sha256']==prov['application_replay']['review']['sha256'], 'Application replay review SHA mismatch')
                    review=json.loads(review_bytes)['models'][link['entry']]
                    require(review==prov['application_replay']['record'], 'Application replay review record changed')
                    require(review['configuration_id']==plan['configuration_id'] and review['reference_result_sha256']==plan['reference_result_sha256'] and review['raw_config_sha256']==plan['raw_config_sha256'], 'Application replay contract mismatch')
                    require(review['replay_mode']=='application' and review['app_replay_match']=='grid' and review['source_reviewed'] is True, 'Application replay method unsupported')
                    for name,digest in review['execution_bindings'].items():
                        require(plan['execution_bindings'].get(name)==digest, 'Application replay source not prospectively bound')
                    require(all(x in prov['command'] for x in ['--replay-mode','application','--app-replay-match','grid','--app-replay-mode','strict']), 'Application replay command missing required options')
                    archive=json.loads(files['application_replay_pass_artifacts.json']);passfiles={}
                    for item in archive['files']:
                        raw=base64.b64decode(item['base64_bytes'],validate=True)
                        require(len(raw)==item['bytes'] and sha(raw)==item['sha256'], 'Application pass artifact SHA/size mismatch')
                        require(item['path'] not in passfiles, 'Application pass duplicate archive path')
                        passfiles[item['path']]=raw
                    if prov['status']=='profile_process_completed':
                        require(prov['application_replay_visible_checks']['all_visible_groups_pass'] is True, 'Application visible checks did not pass')
                        receipts=[(name,json.loads(raw))for name,raw in passfiles.items() if name.endswith('/pass_receipt.json')]
                        require(bool(receipts), 'No application pass receipts')
                        initial=[0]*10;initial[7]=initial[9]=255
                        expected=[8,8,1000]+initial+[160000,15,0,0,0,1,65535,255]+[1,1]+[1515870810]*32
                        for name,receipt in receipts:
                            require(receipt['status']=='all5_property_checks_completed' and receipt['child_returncode']==0, 'Application pass failed original checks')
                            require([r['trial']for r in receipt['STUDY_TRIAL_rows']]==[1,2,3,4,5] and all(r['correct'] is True for r in receipt['STUDY_TRIAL_rows']), 'Application pass five checks absent')
                            require(len(receipt['artifacts'])==5, 'Application pass complete artifact list absent')
                            for artifact in receipt['artifacts']:
                                relative=str(Path(name).parent/Path(artifact['path']).name)
                                raw=passfiles[relative]
                                require(sha(raw)==artifact['sha256'] and len(raw)==220, 'Application property artifact join mismatch')
                                require(list(struct.unpack('<55i',raw))==expected, 'Application property scores/guards mismatch')
                    row['attempt_kind']='source_reviewed_application_replay_property_contract'
                if prov.get('CPU_only_recovery'):
                    require(all(name in files for name in ['parent_profile_provenance.json','counter_derivation_recovery.json','recovery_tool_source.py','half_calibration_snapshot.json']), 'Recovery archive incomplete')
                    parent_bytes=files['parent_profile_provenance.json'];parent=json.loads(parent_bytes)
                    link=prov['counter_derivation_recovery_from'];recovery=json.loads(files['counter_derivation_recovery.json'])
                    require(sha(parent_bytes)==link['sha256']==recovery['parent']['sha256'], 'Recovered parent SHA mismatch')
                    require(sha(files['counter_derivation_recovery.json'])==link['receipt_sha256'], 'Recovery receipt SHA mismatch')
                    require(sha(files['recovery_tool_source.py'])==recovery['tool']['sha256'], 'Recovery source SHA mismatch')
                    require(parent['profile']['returncode']==0 and parent['profile']['stop_reason'] is None, 'Recovery parent process unsuccessful')
                    require(parent['binding_recheck_after']=='unchanged', 'Recovery parent source not rechecked')
                    require(parent['profile']==prov['profile'], 'Recovery changed original execution facts')
                    require(prov.get('new_gpu_processes')==0 and prov.get('new_build_processes')==0, 'Recovery claims a new execution')
                    require(sha(files['counters.csv'])==parent['counters_sha256']==recovery['raw_counter']['sha256'], 'Recovery counter changed')
                    require(sha(files['plan.json'])==parent['plan_sha256'], 'Recovery selected plan changed')
                    require(prov['additional_current_profile_input_bindings']==parent['additional_current_profile_input_bindings'], 'Recovery invented historical input bindings')
                    require(sha(files['collector_source.py'])==parent['collector_sha256'], 'Recovery historical collector mismatch')
                    row['attempt_kind']='CPU_derivation_recovery_no_new_GPU_run'
                if 'counters.csv' in files:
                    if sha(files['counters.csv']) != prov['counters_sha256']:
                        require('counter_snapshot_recovery.json' in files and 'counter_snapshot_at_derivation.csv' in files, 'Counter provenance SHA mismatch')
                        recovery=json.loads(files['counter_snapshot_recovery.json']);snapshot=files['counter_snapshot_at_derivation.csv']
                        require(sha(snapshot)==prov['counters_sha256']==recovery['recovered_snapshot_sha256'], 'Recovered counter snapshot SHA mismatch')
                        require(sha(files['counters.csv'])==recovery['later_counter_sha256'], 'Late counter recovery parent SHA mismatch')
                        require(files['counters.csv'].startswith(snapshot) and len(snapshot)==recovery['recovered_snapshot_bytes'], 'Recovered prefix bytes mismatch')
                        require('derived.json' not in files and prov['status']!='profile_process_completed', 'Late recovery cannot certify accepted derivations')
                        qualifications.append({'profile_id':identity,'qualification':'Timed-out NCU descendants appended after provenance hash. Exact original snapshot recovered by SHA-matching retained byte prefix; late data remains diagnostic only.'})
                if prov.get('allreduce_two_device_method'):
                    try:
                        row['allreduce_method_admission']=validate_allreduce_method(files,root)
                    except Exception as error:
                        state=json.loads((study/'benchmarks'/(plan['benchmark']+'.json')).read_text())['models'][plan['model']]
                        selected_completed=(identity in state.get('profile_ids',[]) and state.get('profiling_status')=='profile_process_completed')
                        require(not selected_completed,'Selected allreduce method is not admitted: '+str(error))
                        row['allreduce_method_admission']={'status':'validation_hold','error':str(error),'positions_publishable':False}
                        qualifications.append({'profile_id':identity,'qualification':'Allreduce method validation hold; no publishable positions or freshly verified primary artifacts: '+str(error)})
                method_error = None
                if prov.get('status')=='not_started_profiler_method_preflight_hold':
                    require(sha(files['profiler_method_preflight.json'])==prov['method_preflight_sha256'],'Preflight hold SHA mismatch')
                    hold=json.loads(files['profiler_method_preflight.json'])
                    require(hold['status']=='rejected' and prov['new_gpu_processes']==prov['new_build_processes']==0 and prov['command']==[],'Invalid no-execution hold')
                    require(not any(n in files for n in ('counters.csv','derived.json','execution.log','build.log')),'No-execution hold contains target artifacts')
                    method_error='method preflight rejected; no target execution'
                elif prov.get('profiler_method') or (prov.get('profiler_version') and prov['profiler_version']!=LEGACY_VERSION):
                    try:
                        method_source=files['profiler_method_validator.py']
                        require(sha(method_source)==prov['profiler_method']['validator_sha256'],'Method validator SHA mismatch')
                        method_module=types.ModuleType('frozen_profiler_method')
                        exec(compile(method_source,'frozen_profiler_method','exec'),method_module.__dict__)
                        method_module.audit_profile(prov,files)
                        row['profiler_method']=prov['profiler_method']
                    except Exception as error:
                        method_error=repr(error)
                if method_error:
                    ledger=json.loads((study/'benchmarks'/(plan['benchmark']+'.json')).read_bytes())
                    require(identity not in ledger['models'][plan['model']].get('profile_ids',[]),'Unapproved method is selected')
                    qualifications.append({'profile_id':identity,'qualification':'Unapproved method retained, no positioning: '+method_error})
                    row['method_validation_status']='unapproved_retained_not_selected'
                if 'derived.json' in files and not method_error:
                    report = json.loads(files['derived.json'])
                    require(report['input_sha256'] == sha(files['counters.csv']), 'Derivation input SHA mismatch')
                    if prov.get('profiler_method'):
                        require(report.get('profiler_method')==prov['profiler_method'],'Derived method mismatch')
                        require(all(x.get('scope_context',{}).get('profiler_method')==prov['profiler_method'] for x in report['records']),'Kernel method mismatch')
                    parser_sha = report['parser_sha256']
                    source = files.get('parser_source.py', snapshots.get(parser_sha))
                    require(source is not None and sha(source) == parser_sha, 'Exact parser source unavailable')
                    if parser_sha not in parsers:
                        module = types.ModuleType('retained_roofline_parser')
                        module.__file__ = str(study / 'tool_snapshots' / (parser_sha+'.py'))
                        exec(compile(source, module.__file__, 'exec'), module.__dict__)
                        parsers[parser_sha] = module
                    parser = parsers[parser_sha]
                    if any(x.get('tensor_convention_enabled') for x in report['records']):
                        require('tensor_mode_decisions.json' in files and 'tensor_mode_loader_source.py' in files,
                                'Enabled tensor projection lacks frozen typed approval')
                        decisions = json.loads(files['tensor_mode_decisions.json'])
                        require(sha(files['tensor_mode_decisions.json']) == prov['tensor_mode_decisions_sha256'], 'Tensor decision SHA mismatch')
                        require(sha(files['tensor_mode_loader_source.py']) == decisions['loader_sha256'], 'Tensor loader SHA mismatch')
                        loader = types.ModuleType('retained_tensor_mode_inventory')
                        loader.__file__ = str(root/'tools/v100_roofline_tensor_modes.py')
                        exec(compile(files['tensor_mode_loader_source.py'],loader.__file__,'exec'),loader.__dict__)
                        expected = loader.required_bindings(plan['benchmark'],plan['model'])
                        require(decisions['profile_input_bindings'] == prov['additional_current_profile_input_bindings'], 'Tensor decision/provenance binding mismatch')
                        for name,digest in expected.items():
                            require(prov['tensor_mode_bindings_before_profile'].get(name) == digest, 'Missing prospective tensor binding')
                            require(decisions['profile_input_bindings'].get(name) == digest, 'Tensor dependency receipt mismatch')
                        require(prov.get('binding_recheck_after') == 'unchanged', 'Tensor dependencies not rechecked')
                        require(len(decisions['records']) == len(report['records']), 'Tensor decision row count mismatch')
                        for saved, item in zip(report['records'],decisions['records']):
                            require(item['identity'] == saved['identity'], 'Tensor decision kernel identity mismatch')
                            if not saved.get('tensor_convention_enabled'): continue
                            decision = item['decision']
                            require(decision['approved'] is True and decision['readonly_bindings'] == expected, 'Unapproved tensor conversion')
                            require(decision['tensor_flops_per_instruction'] == saved['tensor_flops_per_instruction'] == 512, 'Tensor factor mismatch')
                            require(decision['tensor_convention_evidence'] == saved['tensor_convention_evidence'], 'Tensor arithmetic evidence mismatch')
                            require(decision['tensor_precision_mode'] == saved['scope_context']['tensor_precision_mode'] == loader.PRECISION_MODE, 'Tensor mode mismatch')
                            require(saved['scope_context'].get('selected_target_SASS_observed') is False, 'Unobserved target SASS incorrectly promoted')
                    csv_path = Path(temp) / 'counters.csv'
                    csv_path.write_bytes(files['counters.csv'])
                    groups, digest = parser.read_csv(csv_path)
                    require(digest == report['input_sha256'], 'Re-read CSV SHA mismatch')
                    require(len(groups) == report['kernel_count'] == len(report['records']), 'Kernel count mismatch')
                    if prov.get('profile_device_identity_verified'):
                        require(prov['expected_device_pci'] == {'domain':0,'bus':203,'device':0}, 'Expected UUID-to-PCI mapping changed')
                        for group in groups:
                            require(tuple(group['metrics'].get('device__attribute_pci_'+key+'_id') for key in ('domain','bus','device')) == (0,203,0), 'Raw counter PCI row mismatch')
                    parameters = inspect.signature(parser.derive).parameters
                    for group, saved in zip(groups, report['records']):
                        values = dict(half_lanes=saved.get('fp16_lane_factor'),
                                      half_provenance=saved.get('fp16_convention_evidence'),
                                      context=saved.get('scope_context', {}),
                                      tensor_flops_per_instruction=saved.get('tensor_flops_per_instruction'),
                                      tensor_convention_evidence=saved.get('tensor_convention_evidence'))
                        actual = parser.derive(group, **{k:v for k,v in values.items() if k in parameters})
                        require(actual == saved, 'Historical derivation recomputation mismatch: '+str(saved.get('identity')))
                    annotation_path = manifest_path.parent/'annotated.json.gz'
                    if annotation_path.exists():
                        annotation = json.loads(frozen(annotation_path))
                        require(annotation['unannotated_derived_sha256'] == sha(files['derived.json']), 'Annotation parent SHA mismatch')
                        require(len(annotation['records']) == len(report['records']), 'Annotation kernel count mismatch')
                        mutable_scope = {'ROI_stage','semantic_family_in_primary_ROI','sample_phase','ROI_review_sha256','sample_in_primary_timed_invocation','source_capture_binding_qualification'}
                        for saved, labeled in zip(report['records'], annotation['records']):
                            original_scope = saved.get('scope_context', {})
                            revised_scope = labeled.get('scope_context', {})
                            require({k:v for k,v in original_scope.items() if k not in mutable_scope} == {k:v for k,v in revised_scope.items() if k not in mutable_scope}, 'Annotation changed immutable scope facts')
                            numeric = {k:v for k,v in labeled.items() if k not in {'scope_context','ROI_annotation'}}
                            require(numeric == {k:v for k,v in saved.items() if k!='scope_context'}, 'Annotation changed measured or derived fields')
                    row['recomputed_kernels'] = len(groups)
                    row['parser_sha256'] = parser_sha
                if not prov.get('profile_device_identity_verified'):
                    qualifications.append({'profile_id': identity, 'qualification': 'Historical capture lacks explicit raw-row PCI-to-UUID verification; retained device/environment bindings are not a retroactive PCI check.'})
                if prov.get('counter_parse_error'):
                    qualifications.append({'profile_id': identity, 'qualification': 'Counter parse failed: '+prov['counter_parse_error']})
                row['audit_status'] = ('pass_frozen_integrity_method_validation_hold' if row.get('allreduce_method_admission',{}).get('status')=='validation_hold' else 'pass')
            except Exception as error:
                row['audit_status'] = 'failed'
                issues.append({'profile_id': identity, 'error': str(error)})
            attempts.append(row)
        model_count = 0
        ledger_count = 0
        for path in sorted((study / 'benchmarks').glob('*.json')):
            ledger_count += 1
            ledger = json.loads(path.read_text())
            if set(ledger['models']) != {'cuda','cuda_jl','ka','jacc'}:
                issues.append({'ledger': str(path), 'error': 'Incorrect model set'})
            for model, state in ledger['models'].items():
                model_count += 1
                if state.get('profiling_status') == 'roofline_not_applicable_no_device_kernels':
                    try:
                        review_link = state['profile_applicability_review']
                        review_path = root / review_link['path']
                        require(review_path.resolve().is_relative_to(study), 'Applicability review escapes study')
                        review_bytes = review_path.read_bytes()
                        require(sha(review_bytes) == review_link['sha256'], 'Applicability review SHA mismatch')
                        rows = [r for r in json.loads(review_bytes)['rows']
                                if r['benchmark'] == ledger['benchmark'] and r['model'] == model]
                        require(len(rows) == 1, 'Applicability review key is not unique')
                        review = rows[0]
                        parent_sha = review['parent_profile']['sha256']
                        require(parent_sha == review_link['parent_provenance_sha256'] == state['profile_provenance_sha256'], 'Applicability parent join mismatch')
                        parents = []
                        for profile_id in state['profile_ids']:
                            parent_path = root / profile_id / 'profile_provenance.json.gz'
                            data = frozen(parent_path)
                            if sha(data) == parent_sha: parents.append((json.loads(data), json.loads(frozen(parent_path.parent / 'plan.json.gz'))))
                        require(len(parents) == 1, 'Applicability frozen parent unavailable or ambiguous')
                        parent, parent_plan = parents[0]
                        require(parent['profile']['returncode'] == 0 and parent['profile']['stop_reason'] is None, 'Applicability parent execution unsuccessful')
                        require(parent['status'] == 'profile_completed_counter_derivation_failed', 'Unexpected applicability parent status')
                        require(review['five_checked_rows'] is True and len(review['trials']) == 5 and all(t['correct'] for t in review['trials']), 'Applicability five checked trials absent')
                        require(review['ncu_explicit_no_kernels_warning'] is True and review['metrics'] is None, 'Applicability no-kernel evidence absent')
                        require(review['expected_config_sha256'] == parent_plan['raw_config_sha256'] and review['expected_result_sha256'] == parent_plan['reference_result_sha256'], 'Applicability accepted contract mismatch')
                    except Exception as error:
                        issues.append({'benchmark': ledger['benchmark'], 'model': model, 'error': str(error)})
                observation = state.get('kernel_observations')
                if not observation or observation.get('compression') != 'gzip':
                    continue
                try:
                    path = root / observation['path']
                    require(str(path.parent.relative_to(root)) in state['profile_ids'], 'Selected observations belong to a different attempt')
                    selected_provenance=frozen(path.parent/'profile_provenance.json.gz')
                    require(state.get('profile_provenance_sha256')==sha(selected_provenance), 'Selected ledger provenance SHA mismatch')
                    require(sha(path.read_bytes()) == observation['sha256'], 'Selected observation compressed SHA mismatch')
                    require(sha(frozen(path)) == observation['uncompressed_sha256'], 'Selected observation raw SHA mismatch')
                except Exception as error:
                    issues.append({'benchmark': ledger['benchmark'], 'model': model, 'error': str(error)})
        if ledger_count != 537 or model_count != 2148:
            issues.append({'error': 'Unexpected population', 'names': ledger_count, 'models': model_count})
    result = dict(schema_version=1, audited_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  audit_tool_sha256=sha(AUDIT_SOURCE),
                  scope='Frozen manifests, selected contract bytes, historical counter derivations and selected observation hashes. No GPU executions, new correctness certification, complete-ROI or causal proof.',
                  benchmark_names=ledger_count, model_slots=model_count, attempts=len(attempts),
                  recomputed_kernels=sum(x.get('recomputed_kernels',0) for x in attempts),
                  parser_versions=len(parsers), issues=issues, qualifications=qualifications, records=attempts,
                  status_counts=dict(collections.Counter(x['audit_status'] for x in attempts)))
    snapshot=study/'tool_snapshots'/(sha(AUDIT_SOURCE)+'.py.gz')
    if snapshot.exists():require(gzip.decompress(snapshot.read_bytes())==AUDIT_SOURCE, 'Audit source snapshot collision')
    else:snapshot.write_bytes(gzip.compress(AUDIT_SOURCE,mtime=0))
    result['audit_source_snapshot']=str(snapshot.relative_to(root))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix+'.new')
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    temporary.replace(args.output)
    print(json.dumps({k:result[k] for k in ['attempts','recomputed_kernels','parser_versions','status_counts']} | {'issues':len(issues),'qualifications':len(qualifications),'output':str(args.output)}))
    return 1 if issues else 0


if __name__ == '__main__':
    with Path('/tmp/hecbench-roofline-metadata.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        raise SystemExit(main())
