#!/usr/bin/env python3
"""Audit saved roofline evidence and recompute derivations; never launch a GPU.

This checks retained bytes, not the truth of an unproved ROI or old source identity.
The exact hash-bound historical parser is used, rather than silently substituting
the current implementation. Only this repository's retained parser sources run.
"""
import argparse
import collections
import datetime
import fcntl
import gzip
import hashlib
import inspect
import json
from pathlib import Path
import tempfile
import types


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
                if 'counters.csv' in files:
                    if sha(files['counters.csv']) != prov['counters_sha256']:
                        require('counter_snapshot_recovery.json' in files and 'counter_snapshot_at_derivation.csv' in files, 'Counter provenance SHA mismatch')
                        recovery=json.loads(files['counter_snapshot_recovery.json']);snapshot=files['counter_snapshot_at_derivation.csv']
                        require(sha(snapshot)==prov['counters_sha256']==recovery['recovered_snapshot_sha256'], 'Recovered counter snapshot SHA mismatch')
                        require(sha(files['counters.csv'])==recovery['later_counter_sha256'], 'Late counter recovery parent SHA mismatch')
                        require(files['counters.csv'].startswith(snapshot) and len(snapshot)==recovery['recovered_snapshot_bytes'], 'Recovered prefix bytes mismatch')
                        require('derived.json' not in files and prov['status']!='profile_process_completed', 'Late recovery cannot certify accepted derivations')
                        qualifications.append({'profile_id':identity,'qualification':'Timed-out NCU descendants appended after provenance hash. Exact original snapshot recovered by SHA-matching retained byte prefix; late data remains diagnostic only.'})
                if 'derived.json' in files:
                    report = json.loads(files['derived.json'])
                    require(report['input_sha256'] == sha(files['counters.csv']), 'Derivation input SHA mismatch')
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
                row['audit_status'] = 'pass'
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
                observation = state.get('kernel_observations')
                if not observation or observation.get('compression') != 'gzip':
                    continue
                try:
                    path = root / observation['path']
                    require(str(path.parent.relative_to(root)) in state['profile_ids'], 'Selected observations belong to a different attempt')
                    require(sha(path.read_bytes()) == observation['sha256'], 'Selected observation compressed SHA mismatch')
                    require(sha(frozen(path)) == observation['uncompressed_sha256'], 'Selected observation raw SHA mismatch')
                except Exception as error:
                    issues.append({'benchmark': ledger['benchmark'], 'model': model, 'error': str(error)})
        if ledger_count != 537 or model_count != 2148:
            issues.append({'error': 'Unexpected population', 'names': ledger_count, 'models': model_count})
    result = dict(schema_version=1, audited_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  audit_tool_sha256=sha(Path(__file__).read_bytes()),
                  scope='Frozen manifests, selected contract bytes, historical counter derivations and selected observation hashes. No GPU executions, new correctness certification, complete-ROI or causal proof.',
                  benchmark_names=ledger_count, model_slots=model_count, attempts=len(attempts),
                  recomputed_kernels=sum(x.get('recomputed_kernels',0) for x in attempts),
                  parser_versions=len(parsers), issues=issues, qualifications=qualifications, records=attempts,
                  status_counts=dict(collections.Counter(x['audit_status'] for x in attempts)))
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
