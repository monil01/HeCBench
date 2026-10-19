#!/usr/bin/env python3
"""Auditable four-model runner. Configuration supplies benchmark-specific contracts.

Never upgrades native self-checks to direct cross-reference equivalence. Julia
kernel timing is valid only when structured records explicitly exclude JIT.
GPU use is serialized with the study-wide advisory lock.
"""
from __future__ import annotations
import argparse, copy, csv, fcntl, hashlib, json, math, os, re, shutil, sqlite3
import statistics, subprocess, tempfile, struct
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATE = Path(os.environ.get('STATE', ROOT / '.porting-state')) / 'portable_julia'
MODELS = ('cuda', 'cuda_jl', 'ka', 'jacc')
PROTOCOL = 'five_trials_last_four_v1'
TOOL_SOURCE = Path(__file__).read_bytes()
TOOL_SHA256 = hashlib.sha256(TOOL_SOURCE).hexdigest()
EXPORT_MODEL_IDS = {'cuda':'cuda','cuda_jl':'julia','ka':'julia-ka','jacc':'julia-jacc'}

def discover(bench):
    base = ROOT / 'src' / f'{bench}-julia'
    return {'cuda': str(ROOT/'src'/f'{bench}-cuda'),
            'cuda_jl': str(base/f'{bench}-julia-cuda.jl') if (base/f'{bench}-julia-cuda.jl').exists() else str(base),
            'ka': str(base/f'{bench}-julia-KA'), 'jacc': str(base/f'{bench}-julia-JACC')}

def digest_tree(path):
    p = Path(path)
    if not p.exists(): return None
    h = hashlib.sha256()
    for f in sorted(p.rglob('*')):
        if f.is_file() and f.suffix in ('.jl','.cu','.cuh','.h','.cpp','.toml'):
            h.update(str(f.relative_to(p)).encode()); h.update(f.read_bytes())
    return h.hexdigest()

def run_command(command, cwd, env, timeout, log):
    try:
        p = subprocess.run(command, cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout)
        text = p.stdout + '\n' + p.stderr
        code = p.returncode
    except OSError as e:
        text=str(e); code=127
    except subprocess.TimeoutExpired as e:
        text = (e.stdout or b'').decode(errors='replace') if isinstance(e.stdout,bytes) else (e.stdout or '')
        text += '\nTIMEOUT'; code = 124
    log.write_text(text)
    return code, text

def finite_values(v):
    return isinstance(v,list) and bool(v) and all(isinstance(x,(float,int)) and math.isfinite(x) for x in v)

def artifact_values(row):
    artifact = row.get('output_artifact')
    if not artifact: return row.get('values')
    types = {'float32':'f','float64':'d','int32':'i','int64':'q','uint32':'I','uint64':'Q','uint8':'B','int8':'b','uint16':'H','int16':'h'}
    fmt = types.get(artifact.get('dtype'))
    if not fmt or artifact.get('byte_order','little') != 'little': return None
    data = Path(artifact['saved_path']).read_bytes()
    width=struct.calcsize('<'+fmt)
    if len(data) % width or len(data)//width != artifact.get('count'): return None
    return [x[0] for x in struct.iter_unpack('<'+fmt,data)]

_FINITE_ARTIFACT_CACHE = {}
_NUMERIC_COMPARISON_CACHE = {}

def file_sha256(path):
    """Hash every byte without materializing a potentially gigabyte artifact."""
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def stream_float_comparison(a, b, atol, rtol, use_numpy=None):
    """Full scalar contract in bounded chunks; optional vector arithmetic only."""
    fmt = {'float32': ('f', 4), 'float64': ('d', 8)}.get(a.get('dtype'))
    count = a.get('count', 0)
    if (not fmt or not count or a.get('dtype') != b.get('dtype')
            or count != b.get('count')
            or any(x.get('byte_order', 'little') != 'little' for x in (a, b))
            or any(Path(x['saved_path']).stat().st_size != count * fmt[1] for x in (a, b))):
        return {'status': 'unverified', 'reason': 'missing/nonfinite/mismatched numerical values'}
    np = None
    if use_numpy is not False:
        try:
            import numpy as np
        except ImportError:
            pass
    maximum_abs = maximum_rel = 0.0
    valid = True
    checked = 0
    chunk_bytes = 4 * 1024 * 1024
    with Path(a['saved_path']).open('rb') as fa, Path(b['saved_path']).open('rb') as fb:
        while checked < count:
            wanted = min(chunk_bytes, (count - checked) * fmt[1])
            ba, bb = fa.read(wanted), fb.read(wanted)
            if len(ba) != wanted or len(bb) != wanted:
                return {'status': 'unverified', 'reason': 'missing/nonfinite/mismatched numerical values'}
            if np is not None:
                # Python struct values are binary64 even for FP32 artifacts.
                # Promote before subtracting to preserve that exact contract.
                va = np.frombuffer(ba, dtype='<f' + str(fmt[1])).astype(np.float64, copy=False)
                vb = np.frombuffer(bb, dtype='<f' + str(fmt[1])).astype(np.float64, copy=False)
                if not (np.isfinite(va).all() and np.isfinite(vb).all()):
                    return {'status': 'unverified', 'reason': 'missing/nonfinite/mismatched numerical values'}
                with np.errstate(over='ignore', invalid='ignore', under='ignore'):
                    errors = np.abs(va - vb)
                    relative = errors / np.maximum(np.maximum(np.abs(va), atol), 1e-300)
                    valid = valid and bool(np.all(errors <= atol + rtol * np.abs(va)))
                    maximum_abs = max(maximum_abs, float(np.max(errors)))
                    maximum_rel = max(maximum_rel, float(np.max(relative)))
            else:
                for (av,), (bv,) in zip(struct.iter_unpack('<' + fmt[0], ba), struct.iter_unpack('<' + fmt[0], bb)):
                    if not (math.isfinite(av) and math.isfinite(bv)):
                        return {'status': 'unverified', 'reason': 'missing/nonfinite/mismatched numerical values'}
                    error = abs(av - bv)
                    maximum_abs = max(maximum_abs, error)
                    maximum_rel = max(maximum_rel, error / max(abs(av), atol, 1e-300))
                    valid = valid and error <= atol + rtol * abs(av)
            checked += wanted // fmt[1]
    return {'status': 'pass' if valid else 'mismatch', 'checked_values': checked,
            'max_abs_error': maximum_abs, 'max_rel_error': maximum_rel,
            'comparison_engine': 'chunked_numpy_binary64' if np is not None else 'chunked_python_struct_binary64',
            'comparison_engine_version': np.__version__ if np is not None else None,
            'full_value_scan': True, 'chunk_bytes': chunk_bytes}

def exact_input_prefix(ref, got, count):
    if isinstance(count,bool) or not isinstance(count,int) or count<0:raise ValueError('input_prefix_count must be a nonnegative integer')
    if not count:return {'status':'not_requested','meaning':'No automatic serialized-prefix claim; independent input audits may be recorded separately.'}
    a=ref.get('output_artifact');b=got.get('output_artifact')
    widths={'float32':4,'float64':8,'int8':1,'uint8':1,'int16':2,'uint16':2,'int32':4,'uint32':4,'int64':8,'uint64':8}
    if not a or not b or a.get('dtype')!=b.get('dtype') or a.get('dtype') not in widths:
        return {'status':'unverified','reason':'Exact input prefix requires matching saved binary artifact encodings.'}
    if any(x.get('count',0)<count or x.get('byte_order','little')!='little' for x in (a,b)):
        return {'status':'unverified','reason':'Declared input prefix exceeds artifact or uses unsupported byte order.'}
    nbytes=count*widths[a['dtype']]
    with Path(a['saved_path']).open('rb') as fa,Path(b['saved_path']).open('rb') as fb:
        remaining = nbytes
        while remaining:
            wanted = min(4 * 1024 * 1024, remaining)
            pa, pb = fa.read(wanted), fb.read(wanted)
            if len(pa) != wanted or len(pb) != wanted or pa != pb:
                return {'status':'mismatch','reason':'Serialized input prefix bytes differ; output tolerance does not apply.','values':count,'bytes':nbytes}
            remaining -= wanted
    return {'status':'pass','values':count,'bytes':nbytes,'method':'Exact saved prefix bytes; actual kernel input use still requires source/buffer audit.'}

def review_missing_contract_attestations(config, previous_config, previous_result, result_path):
    """Review omitted descriptive attestations without changing execution or criteria."""
    name = config.get('reuse_missing_contract_attestations')
    if not name:
        return None
    if set(config.get('reuse_models', [])) != set(MODELS):
        raise ValueError('Missing-attestation review requires immutable all-four reuse')
    review_path = Path(name)
    if not review_path.is_absolute():
        review_path = ROOT / review_path
    review = json.loads(review_path.read_text())
    if (review.get('source_run_id') != previous_result['run_id']
            or review.get('result_sha256') != hashlib.sha256(result_path.read_bytes()).hexdigest()
            or review.get('config_sha256') != hashlib.sha256(result_path.with_name('config.json').read_bytes()).hexdigest()):
        raise ValueError('Missing-attestation review must bind exact archived result/config')
    if not review.get('reason', '').strip() or not review.get('source_boundary_review', '').strip():
        raise ValueError('Missing-attestation review requires explicit semantic source-boundary review')
    changed = []
    for key in ('inputs_shared_verified', 'timed_regions_equivalent_verified'):
        before, after = previous_config.get(key), config.get(key)
        if before != after:
            if before is not None or after is not True:
                raise ValueError('Only omitted/null attestations may become true; explicit false is not repairable by metadata')
            changed.append(key)
    if not changed or config.get('input_prefix_count', 0) <= 0:
        raise ValueError('Missing-attestation review requires omitted fields and full serialized-input-prefix evidence')
    for category in ('input_evidence', 'timing_evidence'):
        entries = review.get(category)
        if not isinstance(entries, list) or not entries:
            raise ValueError('Missing-attestation review requires source/audit evidence: ' + category)
        for item in entries:
            path = Path(item['path'])
            if not path.is_absolute():
                path = ROOT / path
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != item.get('sha256'):
                raise ValueError('Missing-attestation source/audit evidence hash mismatch: ' + str(path))
    regions = set()
    for model in MODELS:
        data = previous_result['models'].get(model, {})
        rows = data.get('trials', [])
        if (data.get('complete') is not True or data.get('timing_valid') is not True
                or data.get('provenance_valid') is not True or len(rows) != 5):
            raise ValueError('Missing-attestation review cannot repair invalid execution/timing/provenance')
        for row in rows:
            if row.get('correct') is not True or row.get('jit_excluded') is not True or not row.get('timing_region'):
                raise ValueError('Missing-attestation review needs twenty checked, JIT-excluded explicit regions')
            regions.add(row['timing_region'])
    if len(regions) != 1:
        raise ValueError('Missing-attestation review requires identical explicitly recorded timed regions')
    return {'path': str(review_path), 'sha256': hashlib.sha256(review_path.read_bytes()).hexdigest(),
            'source_run_id': previous_result['run_id'], 'updated_fields': changed,
            'source_result_sha256': review['result_sha256'], 'source_config_sha256': review['config_sha256'],
            'reason': review['reason'], 'immutable_execution_and_trials_unchanged': True,
            'scope': 'Omitted descriptive attestations only; all saved output/input-prefix comparisons rechecked, original source/trials/times immutable.'}

def compare_records(ref, got, atol, rtol, input_prefix_count=0, agreement_kind=None):
    if agreement_kind == 'no_numerical_outputs_execution_only':
        # Explicit source-reviewed no-output paths only; absence of an artifact
        # alone is never evidence of numerical agreement.
        def no_output_source(row):
            if row.get('source_empty_kernel') is True:
                return 'empty_kernel'
            if (row.get('source_empty_kernel') is False
                    and row.get('source_no_observable_numerical_outputs') is True
                    and row.get('no_output_reason') == 'original_null_destination'):
                return 'original_null_destination'
            if (row.get('source_empty_kernel') is False
                    and row.get('source_no_device_kernels') is True
                    and row.get('no_numerical_outputs') is True
                    and row.get('no_output_reason') == 'runtime_allocation_lifecycle'
                    and row.get('runtime_api_status_checked') is True
                    and row.get('nonnull_allocations_checked') is True
                    and row.get('allocation_lifecycle_checked') is True):
                return 'runtime_allocation_lifecycle'
            if (row.get('source_empty_kernel') is False
                    and row.get('source_no_observable_numerical_outputs') is True
                    and row.get('no_output_reason') == 'source_final_pointer_assertion_only'
                    and row.get('source_final_pointer_assertion_checked') is True
                    and row.get('pointer_graph_checked') is True
                    and row.get('device_completion_checked') is True
                    and row.get('nonnull_final_pointer_checked') is True
                    and row.get('pointer_storage_bits') == 64):
                return 'source_final_pointer_assertion_only'
            if (row.get('source_empty_kernel') is False
                    and row.get('source_no_observable_numerical_outputs') is True
                    and row.get('no_output_reason') == 'source_device_assertion_only'
                    and row.get('device_assertions_present') is True
                    and row.get('successful_assertion_kernels_checked') is True
                    and row.get('expected_device_assertion_failure_checked') is True
                    and row.get('expected_device_assertion_status') == 710
                    and row.get('observed_device_assertion_status') == 710
                    and row.get('assertion_failure_context_isolated') is True
                    and row.get('device_completion_checked') is True):
                return 'source_device_assertion_only'
            return None
        reasons = [no_output_source(row) for row in (ref, got)]
        valid = input_prefix_count == 0 and reasons[0] is not None and reasons[0] == reasons[1] and all(
            row.get('execution_checked') is True
            and row.get('correct') is True and row.get('checked_values') == 0
            and not row.get('output_artifact') and row.get('values') in (None, [])
            and row.get('numerical_outputs') in (None, [])
            and row.get('numeric_outputs') in (None, [])
            for row in (ref, got))
        return {'status': 'execution_only_pass' if valid else 'unverified',
                'checked_values': 0, 'max_abs_error': None, 'max_rel_error': None,
                'numerical_agreement_claim': False,
                'source_no_output_reason': reasons[0] if valid else None,
                'meaning': 'Reviewed source path has no observable numerical outputs; checked launch/completion only.'}

    prefix_check=exact_input_prefix(ref,got,input_prefix_count)
    if prefix_check['status'] not in ('pass','not_requested'):return {'status':prefix_check['status'],'reason':prefix_check['reason'],'input_prefix_check':prefix_check}
    a=ref.get('output_artifact'); b=got.get('output_artifact')
    if a and b and any(a.get(k)!=b.get(k) for k in ('dtype','count','shape')):
        return {'status':'mismatch','reason':'artifact dtype/count/shape mismatch'}
    # Identical saved bytes need one finite-validation scan, not repeated
    # multi-million-element Python lists for every dual-reference/trial pair.
    if a and b and a.get('sha256') and a['sha256']==b.get('sha256'):
        if file_sha256(a['saved_path'])!=a['sha256'] or file_sha256(b['saved_path'])!=b['sha256']:
            return {'status':'unverified','reason':'saved artifact content hash changed'}
        types={'float32':('f',4),'float64':('d',8),'int8':('b',1),'uint8':('B',1),'int16':('h',2),'uint16':('H',2),'int32':('i',4),'uint32':('I',4),'int64':('q',8),'uint64':('Q',8)}
        info=types.get(a.get('dtype'))
        if not info or a.get('byte_order','little')!='little' or b.get('byte_order','little')!='little':
            return {'status':'unverified','reason':'unsupported artifact storage'}
        expected=a.get('count',0)*info[1]
        if not a.get('count') or Path(a['saved_path']).stat().st_size!=expected or Path(b['saved_path']).stat().st_size!=expected:
            return {'status':'unverified','reason':'artifact count/byte-size mismatch'}
        key=(a['sha256'],a.get('dtype'),a['count'])
        if key not in _FINITE_ARTIFACT_CACHE:
            finite=True
            if a.get('dtype') in ('float32','float64'):
                with Path(a['saved_path']).open('rb') as stream:
                    while chunk := stream.read(4 * 1024 * 1024):
                        if not all(math.isfinite(x[0]) for x in struct.iter_unpack('<'+info[0],chunk)):
                            finite=False
                            break
            _FINITE_ARTIFACT_CACHE[key]=finite
        if not _FINITE_ARTIFACT_CACHE[key]:
            return {'status':'unverified','reason':'nonfinite artifact values'}
        return {'status':'pass','checked_values':a['count'],'max_abs_error':0,'max_rel_error':0,'input_prefix_check':prefix_check}
    # Deterministic outputs may repeat across all five independently executed
    # trials/models. Reuse only the mathematical comparison, after rehashing
    # both actual files; every trial and source/input check remains separate.
    cache_key = None
    if a and b and a.get('sha256') and b.get('sha256'):
        for artifact in (a, b):
            if file_sha256(artifact['saved_path']) != artifact['sha256']:
                return {'status':'unverified','reason':'saved artifact content hash changed'}
        metadata = [{k: artifact.get(k, 'little' if k == 'byte_order' else None)
                     for k in ('sha256','dtype','count','shape','byte_order')}
                    for artifact in (a, b)]
        cache_key = json.dumps([metadata, atol, rtol, input_prefix_count, agreement_kind], sort_keys=True)
        if cache_key in _NUMERIC_COMPARISON_CACHE:
            comparison = copy.deepcopy(_NUMERIC_COMPARISON_CACHE[cache_key])
            comparison['comparison_computation'] = 'reused_after_full_artifact_pair_rehash'
            comparison['input_prefix_check'] = prefix_check
            return comparison
    large_float = (a and b and a.get('dtype') in ('float32', 'float64')
                   and a.get('sha256') and b.get('sha256')
                   and a.get('count', 0) * (4 if a['dtype'] == 'float32' else 8) >= 64 * 1024 * 1024)
    comparison = (stream_float_comparison(a, b, atol, rtol) if large_float
                  else compare(artifact_values(ref),artifact_values(got),atol,rtol))
    if cache_key is not None and comparison.get('status') == 'pass':
        _NUMERIC_COMPARISON_CACHE[cache_key] = copy.deepcopy(comparison)
        comparison['comparison_computation'] = 'computed_from_full_saved_values'
    comparison['input_prefix_check']=prefix_check
    return comparison

def save_artifacts(rows, cwd, out, model):
    for row in rows:
        artifact=row.get('output_artifact')
        if not artifact: continue
        path=Path(artifact['path'])
        if not path.is_absolute(): path=cwd/path
        saved=out/f"{model}_trial{row.get('trial')}_{path.name}"
        shutil.copyfile(path,saved)
        artifact['saved_path']=str(saved)
        artifact['sha256']=file_sha256(saved)

def preserve_rejected_records(text, cwd, out, model, events, returncode, reason):
    """Keep diagnostics from failed execution; never feed them to acceptance."""
    def diagnostic_json(value):
        if isinstance(value,float) and not math.isfinite(value):return repr(value)
        if isinstance(value,list):return [diagnostic_json(x) for x in value]
        if isinstance(value,dict):return {k:diagnostic_json(v) for k,v in value.items()}
        return value
    rejected=[]
    for line in text.splitlines():
        if not line.startswith('STUDY_TRIAL '):continue
        try:
            row=json.loads(line[len('STUDY_TRIAL '):])
            if not isinstance(row,dict):raise ValueError('Structured trial must be an object')
        except (ValueError,TypeError) as error:
            events.append({'model':model,'phase':'rejected_evidence_parse','error':str(error),'raw_record':line})
            continue
        try:
            save_artifacts([row],cwd,out,model)
        except (OSError,ValueError,TypeError,KeyError) as error:
            row['rejected_artifact_capture_error']=str(error)
        row['rejection_context']={'returncode':returncode,'reason':reason,
            'acceptance_eligible':False,'raw_record':line,
            'nonfinite_diagnostic_scalars':'Represented as strings; original tokens remain in raw_record/log. Artifact bytes are unchanged.'}
        rejected.append(diagnostic_json(row))
    return rejected

def preserve_unreported_failure_artifacts(cwd, out, model, events, initial_binaries):
    """Retain generated .bin diagnostics even if a checker throws before a trial row.

    No dtype/shape/correctness is inferred. Unchanged runtime inputs are excluded.
    These files never participate in accepted trials, comparisons or timings.
    """
    folder=out/(model+'_unreported_failure_artifacts')
    for source in sorted(cwd.rglob('*.bin')):
        if not source.is_file() or source.is_symlink():continue
        relative=source.relative_to(cwd)
        try:
            sha=file_sha256(source)
            if initial_binaries.get(str(relative))==sha:continue
            destination=folder/relative
            destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(source,destination)
            events.append({'model':model,'phase':'unreported_failure_artifact_capture',
                'relative_path':str(relative),'saved_path':str(destination),
                'sha256':sha,'size_bytes':destination.stat().st_size,
                'acceptance_eligible':False,'dtype':None,'shape':None,
                'scope':'Generated binary retained after failed execution/evidence parse; no structured trial claim. Historical missing captures are not recovered.'})
        except OSError as error:
            events.append({'model':model,'phase':'unreported_failure_artifact_capture_error',
                'relative_path':str(relative),'error':str(error),'acceptance_eligible':False})

def compare(ref, got, atol, rtol):
    if not finite_values(ref) or not finite_values(got) or len(ref)!=len(got):
        return {'status':'unverified','reason':'missing/nonfinite/mismatched numerical values'}
    ae=[abs(a-b) for a,b in zip(ref,got)]
    re_=[e/max(abs(a),atol,1e-300) for a,e in zip(ref,ae)]
    ok=all(e<=atol+rtol*abs(a) for a,e in zip(ref,ae))
    return {'status':'pass' if ok else 'mismatch','checked_values':len(ae),'max_abs_error':max(ae),'max_rel_error':max(re_)}

def review_missing_timing_region(config,reuse_result,reuse_config,reuse_path,results):
    """Metadata-only review, never changes trial rows, time samples or acceptance."""
    name=config.get('reuse_timing_region_metadata')
    if not name:return None
    if reuse_result is None or set(config.get('reuse_models',[]))!=set(MODELS):
        raise ValueError('Timing metadata review requires explicit all-four-model reuse')
    if reuse_result.get('timing_region_metadata_adjudication'):
        raise ValueError('Do not chain timing metadata adjudications')
    if reuse_config.get('timed_regions_equivalent_verified') is not True or not reuse_config.get('input_contract',{}).get('timing'):
        raise ValueError('Original measurement must already declare its timing contract/equivalence')
    path=Path(name);path=path if path.is_absolute() else ROOT/path
    raw=path.read_bytes();review=json.loads(raw)
    if review.get('source_run_id')!=reuse_result['run_id'] or review.get('result_sha256')!=hashlib.sha256(reuse_path.read_bytes()).hexdigest() or review.get('config_sha256')!=hashlib.sha256(reuse_path.with_name('config.json').read_bytes()).hexdigest():
        raise ValueError('Timing metadata source/config hash mismatch')
    region=review.get('timing_region')
    if review.get('metadata_only') is not True or not isinstance(region,str) or not region.strip() or not review.get('reason','').strip():
        raise ValueError('Timing metadata review needs a nonempty descriptor and reason')
    model_reviews=review.get('model_reviews',{})
    if set(model_reviews)!=set(MODELS):raise ValueError('Timing source review must cover all four models')
    evidence=review.get('evidence',[])
    if not evidence:raise ValueError('Timing metadata review requires hashed source evidence')
    for item in evidence:
        f=Path(item['path']);f=f if f.is_absolute() else ROOT/f
        if not f.is_file() or file_sha256(f)!=item.get('sha256'):
            raise ValueError('Timing metadata source evidence mismatch: '+str(f))
    for model in MODELS:
        data=results.get(model,{})
        if data.get('reused_from_run_id')!=reuse_result['run_id'] or len(data.get('trials',[]))!=5:
            raise ValueError('Timing metadata review requires twenty unchanged reused trials')
        if any(row.get('timing_region') is not None for row in data['trials']):
            raise ValueError('Timing metadata review only fills uniformly missing descriptors')
        mr=model_reviews[model]
        if not mr.get('start_boundary','').strip() or not mr.get('completion_boundary','').strip() or mr.get('setup_reset_capture_outside_roi') is not True:
            raise ValueError('Missing timing source review boundaries: '+model)
    return {'path':str(path.relative_to(ROOT)),'sha256':hashlib.sha256(raw).hexdigest(),
            'source_run_id':reuse_result['run_id'],'source_result_sha256':review['result_sha256'],
            'source_config_sha256':review['config_sha256'],'effective_timing_region':region,
            'reason':review['reason'],'trial_rows_and_samples_unchanged':True,
            'scope':'Postmeasurement descriptive annotation from predeclared contract and reviewed unchanged sources; no new execution, numerical acceptance, time unit or measured region.'}

def parse_records(text, spec, external_trial=None):
    rows=[]
    for line in text.splitlines():
        if line.startswith('STUDY_TRIAL '): rows.append(json.loads(line[len('STUDY_TRIAL '):]))
    if rows: return rows
    if external_trial is None: return []
    # Legacy reference adapters must declare the actual unit, timing region,
    # JIT handling and native verifier audit. No generic PASS inference.
    t=None
    if spec.get('time_regex'):
        m=re.search(spec['time_regex'],text)
        if m: t=float(m.group(1))*float(spec.get('time_scale_seconds',1))
    correct=None
    if spec.get('native_check_audited') and spec.get('pass_regex'):
        correct=bool(re.search(spec['pass_regex'],text)) and not bool(re.search(spec.get('fail_regex',r'(?i)\bFAIL(?:ED)?\b'),text))
    values=[]
    for pattern in spec.get('value_regexes',[]):
        for match in re.finditer(pattern,text): values.extend(float(x) for x in match.groups())
    return [{'trial':external_trial,'time_seconds':t,'correct':correct,'values':values,
             'checked_values':spec.get('checked_values'), 'jit_excluded':spec.get('jit_excluded',False),
             'timing_region':spec.get('timing_region'),'internal_repeats':spec.get('internal_repeats')}]

def export_global_csv():
    dbpath=STATE/'results.db'
    if not dbpath.exists(): return
    with (STATE/'export.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        db=sqlite3.connect(dbpath,timeout=60)
        records=[json.loads(x[0]) for x in db.execute('SELECT result_json FROM runs ORDER BY run_id')]
        db.close()
        target=STATE/'timings.csv'; tmp=STATE/'timings.csv.tmp'
        with tmp.open('w') as f:
            w=csv.writer(f)
            w.writerow(['run_id','benchmark','configuration_id','model','trial','time_seconds','correct','jit_excluded','mean_last_four_seconds','matched_performance_valid','reused_from_run_id'])
            for result in records:
                for model,data in result['models'].items():
                    for row in data.get('trials',[]):
                        w.writerow([result['run_id'],result['benchmark'],result.get('configuration_id'),EXPORT_MODEL_IDS[model],row.get('trial'),row.get('time_seconds'),row.get('correct'),row.get('jit_excluded'),data.get('mean_seconds'),result.get('matched_performance_valid'),data.get('reused_from_run_id')])
        os.replace(tmp,target)
        alias=STATE/'verification.db'
        if not alias.exists(): alias.symlink_to('results.db')

def verify(config):
    prefix_count=config.get('input_prefix_count',0)
    if isinstance(prefix_count,bool) or not isinstance(prefix_count,int) or prefix_count<0:raise ValueError('input_prefix_count must be a nonnegative integer')
    if any(k in config for k in ('reuse_verified_models','reuse_from_result')):
        raise ValueError('Unsupported top-level reuse keys: use reuse_result and reuse_models; unknown reuse requests must not silently rerun references.')
    if any('reuse_run' in spec for spec in config.get('models',{}).values()):
        raise ValueError('Unsupported per-model reuse_run: use top-level reuse_result and reuse_models with strict source/config checks.')
    bench=config['benchmark']; run_id=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'_'+bench
    out=STATE/'runs'/run_id; out.mkdir(parents=True)
    (out/'config.json').write_text(json.dumps(config,indent=2))
    (out/'verifier_source.py').write_bytes(TOOL_SOURCE)
    env=os.environ.copy(); env['CUDA_VISIBLE_DEVICES']=str(config.get('gpu',6))
    paths=discover(bench); results={}; events=[]
    timeout=config.get('timeout',300)
    hardware=subprocess.run(['nvidia-smi','--query-gpu=index,name,uuid,compute_cap,driver_version','--format=csv,noheader'],capture_output=True,text=True).stdout
    (out/'hardware.txt').write_text(hardware)
    reuse_result=None
    reuse_config=None
    reuse_configuration_transition=None
    prefix_validation_transition=None
    contract_attestation_review=None
    if config.get('reuse_result'):
        reuse_path=Path(config['reuse_result'])
        reuse_result=json.loads(reuse_path.read_text())
        reuse_config=json.loads((reuse_path.parent/'config.json').read_text())
        if reuse_result['benchmark']!=bench or reuse_result['protocol']!=PROTOCOL or reuse_result.get('input_contract')!=config.get('input_contract'):
            raise ValueError('reuse benchmark/protocol/input contract mismatch')
        contract_attestation_review=review_missing_contract_attestations(config,reuse_config,reuse_result,reuse_path)
        for key in ('atol','rtol','gpu','inputs_shared_verified','timed_regions_equivalent_verified'):
            if reuse_config.get(key)!=config.get(key):
                if contract_attestation_review and key in contract_attestation_review['updated_fields']:continue
                raise ValueError('reuse config mismatch: '+key)
        old_prefix_count=reuse_config.get('input_prefix_count',0)
        if old_prefix_count!=prefix_count:
            reason=config.get('reuse_prefix_revalidation_reason')
            if prefix_count<old_prefix_count or not isinstance(reason,str) or not reason.strip():
                raise ValueError('Strengthened input-prefix validation on reused captures requires a nonempty reuse_prefix_revalidation_reason; weakening is forbidden.')
            prefix_validation_transition={'source_run_id':reuse_result['run_id'],'old_count':old_prefix_count,'new_count':prefix_count,'reason':reason.strip(),'scope':'Saved artifacts rechecked under stronger byte-prefix acceptance; archived configs and executions unchanged.'}
        if reuse_config.get('configuration_id')!=config.get('configuration_id'):
            reason=config.get('cross_configuration_reuse_reason')
            if config.get('allow_cross_configuration_reuse') is not True or not isinstance(reason,str) or not reason.strip():
                raise ValueError('reuse configuration_id differs: require allow_cross_configuration_reuse:true and a nonempty cross_configuration_reuse_reason')
            if not reuse_config.get('configuration_id') or not config.get('configuration_id'):
                raise ValueError('cross-configuration reuse requires named old and new configuration IDs')
            reuse_configuration_transition={'source_run_id':reuse_result['run_id'],
                'old_configuration_id':reuse_config['configuration_id'],
                'new_configuration_id':config['configuration_id'],'reason':reason.strip(),
                'unchanged_models':[]}

    if config.get('reuse_missing_contract_attestations') and reuse_result is None:
        raise ValueError('Missing-attestation review requires an archived reuse_result')

    if config.get('reuse_timing_region_metadata') and (reuse_result is None or set(config.get('reuse_models',[]))!=set(MODELS)):
        raise ValueError('Timing metadata annotation requires all-four reuse before any execution')

    for model in MODELS:
        spec=config.get('models',{}).get(model)
        if not spec:
            results[model]={'status':'missing','trials':[]}; continue
        cwd=Path(spec.get('cwd',paths[model])); cwd= cwd if cwd.is_absolute() else ROOT/cwd
        source_cwd=cwd
        adapter_file_hashes={}
        for arg in spec.get('command',[])+spec.get('build',[])+spec.get('source_dependencies',[]):
            f=Path(arg)
            if not f.is_absolute(): f=source_cwd/f
            if f.is_file() and f.suffix in ('.jl','.cu','.cuh','.h','.hpp','.cpp','.toml'):
                adapter_file_hashes[str(f)]=file_sha256(f)
        # Opt in without changing historical reuse bindings: declared include,
        # fixture and runtime-asset files may use arbitrary extensions.
        if spec.get('bind_all_declared_dependencies', False):
            for name in [*spec.get('source_dependencies', []), *spec.get('runtime_assets', [])]:
                f=Path(name)
                if not f.is_absolute(): f=source_cwd/f
                f=f.resolve()
                if not f.is_file():
                    raise FileNotFoundError('Declared dependency is missing: '+str(f))
                adapter_file_hashes[str(f)]=file_sha256(f)
        source_hash=digest_tree(paths[model])
        if reuse_result and model in config.get('reuse_models',['cuda','cuda_jl','ka']):
            old=reuse_result['models'].get(model,{})
            if not old.get('complete') or old.get('source_hash')!=source_hash or reuse_config.get('models',{}).get(model)!=spec:
                raise ValueError('reuse source/config/incomplete mismatch: '+model)
            old_files=old.get('adapter_file_hashes')
            reviewed_changes=[]
            if old_files is not None and old_files!=adapter_file_hashes:
                allowed=config.get('reuse_reviewed_dependency_changes',[])
                for filename in set(old_files)|set(adapter_file_hashes):
                    previous=old_files.get(filename);current=adapter_file_hashes.get(filename)
                    if previous==current:continue
                    review=next((r for r in allowed if Path(r['path']).resolve()==Path(filename).resolve() and r.get('old_sha256')==previous and r.get('new_sha256')==current and r.get('reason','').strip()),None)
                    if review is None:raise ValueError('reuse adapter dependency changed without exact-hash review: '+model+' '+filename)
                    reviewed_changes.append(review)

            if old_files is None and not config.get('reuse_legacy_adapter_unchanged_verified'):
                raise ValueError('legacy bundle lacks individual adapter hashes: explicitly audit unchanged adapters first')
            results[model]=dict(old)
            results[model]['reused_from_run_id']=reuse_result['run_id']
            results[model]['adapter_file_hashes']=adapter_file_hashes
            if reviewed_changes:results[model]['reviewed_dependency_changes']=reviewed_changes
            if reuse_configuration_transition is not None:
                reuse_configuration_transition['unchanged_models'].append(EXPORT_MODEL_IDS[model])
                results[model]['reused_from_configuration_id']=reuse_config['configuration_id']
            events.append({'model':model,'phase':'reuse_verified_five_trial_bundle','source_run_id':reuse_result['run_id'],'legacy_adapter_audit':old_files is None,'configuration_transition':reuse_configuration_transition})
            continue
        adapter_hash=digest_tree(source_cwd)
        temp=tempfile.TemporaryDirectory(prefix=f'hecbench-{model}-')
        cwd=Path(temp.name)
        for asset in spec.get('runtime_assets',[]):
            src=source_cwd/asset
            dst=cwd/asset
            dst.parent.mkdir(parents=True,exist_ok=True)
            if src.is_dir(): shutil.copytree(src,dst)
            else: shutil.copy2(src,dst)
        initial_binaries={str(f.relative_to(cwd)):file_sha256(f) for f in cwd.rglob('*.bin') if f.is_file() and not f.is_symlink()}
        model_env=env.copy()
        model_env['STUDY_OUTPUT_DIR']=str(cwd)
        def resolve_args(args, building=False):
            resolved=[]
            for i,arg in enumerate(args):
                candidate=source_cwd/arg
                if i>0 and (i==0 or args[i-1] != '-o') and candidate.is_file() and candidate.suffix in ('.jl','.cu','.cpp','.c','.h','.cuh'):
                    resolved.append(str(candidate))
                elif i==0 and arg.startswith('./') and not spec.get('build'):
                    resolved.append(str(candidate))
                else: resolved.append(arg)
            return resolved
        provenance_valid=True
        provenance_checks=[]
        def sources_stable():
            nonlocal provenance_valid
            stable=digest_tree(paths[model])==source_hash and all(Path(name).is_file() and file_sha256(name)==sha for name,sha in adapter_file_hashes.items())
            provenance_valid=provenance_valid and stable
            provenance_checks.append({'unchanged':stable})
            return stable
        build_ok=sources_stable()
        if build_ok and spec.get('build'):
            # ARCH supplied from hardware capability, never inherited sm120.
            query=subprocess.run(['nvidia-smi','-i',env['CUDA_VISIBLE_DEVICES'],'--query-gpu=compute_cap','--format=csv,noheader'],capture_output=True,text=True)
            capabilities={line.strip().replace('.','') for line in query.stdout.splitlines() if line.strip()}
            if query.returncode or len(capabilities)!=1:
                raise ValueError('Native build requires one known shared compute capability for selected GPUs: '+query.stdout.strip())
            arch='sm_'+next(iter(capabilities))
            command=resolve_args([s.replace('{arch}',arch) for s in spec['build']],True)
            code,text=run_command(command,cwd,model_env,timeout,out/f'{model}_build.log')
            events.append({'model':model,'phase':'build','command':command,'returncode':code,'source_hash':source_hash})
            stable_after_build=sources_stable()
            build_ok=code==0 and stable_after_build
            if code==0 and not build_ok:events.append({'model':model,'phase':'source_provenance','status':'invalid','reason':'source changed during build'})
        trials=[]
        rejected_trials=[]
        if build_ok:
            count=1 if spec.get('structured',False) else 5
            with open('/tmp/hecbench-julia-study-gpu.lock','w') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX)
                for n in range(1,count+1):
                    if not sources_stable():
                        events.append({'model':model,'phase':'source_provenance','status':'invalid','reason':'source changed while queued before execution'});break
                    command=resolve_args(spec['command'])
                    code,text=run_command(command,cwd,model_env,timeout,out/f'{model}_run{n}.log')
                    events.append({'model':model,'phase':'execution_device_jit','command':spec['command'],'returncode':code,'source_hash':source_hash})
                    stable_after_execution=sources_stable()
                    if not stable_after_execution:
                        events.append({'model':model,'phase':'source_provenance','status':'invalid','reason':'source changed during execution'})
                    if code or not stable_after_execution:
                        reason='execution_nonzero_exit' if code else 'source_provenance_invalid'
                        rejected_trials.extend(preserve_rejected_records(text,cwd,out,model,events,code,reason))
                        preserve_unreported_failure_artifacts(cwd,out,model,events,initial_binaries)
                        break
                    try:
                        rows=parse_records(text,spec,None if count==1 else n)
                        save_artifacts(rows,cwd,out,model)
                        trials.extend(rows)
                    except (ValueError, OSError, KeyError) as error:
                        events.append({'model':model,'phase':'evidence_parse','error':str(error)})
                        rejected_trials.extend(preserve_rejected_records(text,cwd,out,model,events,code,'accepted_evidence_parse_failed'))
                        preserve_unreported_failure_artifacts(cwd,out,model,events,initial_binaries)
                        break
        if temp: temp.cleanup()
        indices=[r.get('trial') for r in trials]
        complete=provenance_valid and indices==[1,2,3,4,5]
        correct=complete and all(r.get('correct') is True for r in trials)
        times=[r.get('time_seconds') for r in trials]
        timing_valid=complete and all(isinstance(t,(int,float)) and math.isfinite(t) and t>0 for t in times) and all(r.get('jit_excluded') is True for r in trials)
        results[model]={'status':'native_check_pass' if correct else 'provenance_invalid' if not provenance_valid else 'unverified','trials':trials if provenance_valid else [],
                        'rejected_trials':rejected_trials,'rejected_trial_policy':'Diagnostic only; excluded from trials, completeness, numerical comparisons, timing means and verification.',
                        'provenance_valid':provenance_valid,'provenance_checks':provenance_checks,
                        'complete':complete,'timing_valid':timing_valid,'mean_seconds':statistics.mean(times[1:]) if timing_valid else None,
                        'source_hash':source_hash,'source_path':paths[model],'adapter_hash':adapter_hash,'adapter_source_path':str(source_cwd),'adapter_file_hashes':adapter_file_hashes,'export_model_id':EXPORT_MODEL_IDS[model], 'verification_method':spec.get('verification_method','unspecified'),
                        'native_check_description':spec.get('native_check_description'), 'limitations':spec.get('limitations',[])}
    comparisons={}
    for model in ('ka','jacc'):
        comparisons[model]={}
        for ref in ('cuda','cuda_jl'):
            a=results[ref].get('trials',[]); b=results[model].get('trials',[])
            checks=[compare_records(x,y,config.get('atol',1e-6),config.get('rtol',1e-5),config.get('input_prefix_count',0),config.get('numerical_agreement_kind')) for x,y in zip(a,b)]
            if len(checks)==5 and all(x['status']=='execution_only_pass' for x in checks):
                status='execution_only_pass'
            elif len(checks)==5 and all(x['status']=='pass' for x in checks):
                status='direct_artifact_pass' if all(x.get('output_artifact') and y.get('output_artifact') for x,y in zip(a,b)) else 'numeric_summary_pass'
            elif any(x['status']=='mismatch' for x in checks): status='mismatch'
            elif results[ref]['status']=='native_check_pass' and results[model]['status']=='native_check_pass': status='independent_native_checks_only'
            else: status='unverified'
            comparisons[model][ref]={'status':status,'trials':checks}
    ref_a=results['cuda'].get('trials',[])
    ref_b=results['cuda_jl'].get('trials',[])
    reference_checks=[compare_records(x,y,config.get('atol',1e-6),config.get('rtol',1e-5),config.get('input_prefix_count',0),config.get('numerical_agreement_kind')) for x,y in zip(ref_a,ref_b)]
    reference_disagreement=any(x['status']=='mismatch' for x in reference_checks)
    reference_agreement=(len(reference_checks)==5 and all(x['status']==('execution_only_pass' if config.get('numerical_agreement_kind')=='no_numerical_outputs_execution_only' else 'pass') for x in reference_checks))
    variant_verified={}
    for model in ('ka','jacc'):
        comparisons_ok=all(v['status'] in ('direct_artifact_pass','numeric_summary_pass','execution_only_pass') for v in comparisons[model].values())
        variant_verified[model]=reference_agreement and comparisons_ok and all(results[m]['status']=='native_check_pass' for m in ('cuda','cuda_jl',model))
    timing_metadata_adjudication=review_missing_timing_region(config,reuse_result,reuse_config,
        reuse_path if reuse_result is not None else None,results)
    regions={r.get('timing_region') for v in results.values() for r in v.get('trials',[])}
    if timing_metadata_adjudication:
        regions={timing_metadata_adjudication['effective_timing_region']}
        events.append({'model':'shared','phase':'metadata_only_timing_region_review',
                       'review':timing_metadata_adjudication,'new_GPU_executions':0})
    matched=bool(config.get('inputs_shared_verified')) and bool(config.get('timed_regions_equivalent_verified')) and len(regions)==1 and None not in regions and all(v.get('timing_valid') and v.get('status')=='native_check_pass' for v in results.values()) and reference_agreement and all(variant_verified.values())
    result={'run_id':run_id,'benchmark':bench,'configuration_id':config.get('configuration_id'),'tooling_sha256':TOOL_SHA256,'database_path':str(STATE/'results.db'),'database_alias':str(STATE/'verification.db'),'protocol':PROTOCOL,'excluded_trial':1,'retained_trials':[2,3,4,5],
            'contract_attestation_metadata_adjudication':contract_attestation_review,'numerical_agreement_kind':config.get('numerical_agreement_kind','declared_numeric_artifact_contract'),'numerical_agreement_claim':config.get('numerical_agreement_kind') not in ('no_numerical_outputs_execution_only','property_equivalence_not_identical_raw_values'),'raw_output_identity_claim':False if config.get('raw_output_identity_claim') is False or config.get('numerical_agreement_kind') in ('no_numerical_outputs_execution_only','property_equivalence_not_identical_raw_values') else None,'input_prefix_count':config.get('input_prefix_count',0),'model_id_mapping':EXPORT_MODEL_IDS,'models':results,'comparisons':comparisons,'variant_verified':variant_verified,'reference_agreement':reference_agreement,'reference_disagreement':reference_disagreement,'reference_checks':reference_checks,'matched_performance_valid':matched,'input_contract':config.get('input_contract'),
            'ontology':config.get('ontology'), 'token_accounting':config.get('token_accounting',{'status':'unknown','input_tokens':None,'output_tokens':None}),
            'attempt_accounting':config.get('attempt_accounting'),'timing_region_metadata_adjudication':timing_metadata_adjudication,'prefix_validation_transition':prefix_validation_transition,'reuse_configuration_transition':reuse_configuration_transition,'events':events}
    (out/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    db=sqlite3.connect(STATE/'results.db',timeout=60)
    db.execute('PRAGMA journal_mode=WAL')
    db.execute('CREATE TABLE IF NOT EXISTS runs(run_id TEXT PRIMARY KEY,benchmark TEXT,result_json TEXT)')
    db.execute('INSERT INTO runs VALUES(?,?,?)',(run_id,bench,json.dumps(result,allow_nan=False))); db.commit(); db.close()
    with (out/'timings.csv').open('w') as f:
        w=csv.writer(f); w.writerow(['run_id','benchmark','configuration_id','model','trial','time_seconds','correct','jit_excluded'])
        for model,v in results.items():
            for row in v.get('trials',[]): w.writerow([run_id,bench,config.get('configuration_id'),EXPORT_MODEL_IDS[model],row.get('trial'),row.get('time_seconds'),row.get('correct'),row.get('jit_excluded')])
    export_global_csv()
    print(json.dumps({'result':str(out/'result.json'),'matched_performance_valid':matched,'comparisons':comparisons}))
    requested=[m for m in ('ka','jacc') if m in config.get('models',{})]
    return 0 if requested and all(variant_verified[m] for m in requested) else 2

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('benchmark',nargs='?'); p.add_argument('--config',type=Path); p.add_argument('--discover',action='store_true'); p.add_argument('--export',action='store_true')
    a=p.parse_args(); STATE.mkdir(parents=True,exist_ok=True)
    if a.export: export_global_csv(); return 0
    if a.discover: print(json.dumps(discover(a.benchmark),indent=2)); return 0
    if not a.config: p.error('--config required: benchmark-specific inputs/checker/timing contracts cannot safely be inferred')
    config=json.loads(a.config.read_text()); return verify(config)
if __name__=='__main__': raise SystemExit(main())
