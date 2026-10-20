#!/usr/bin/env python3
from pathlib import Path
import array
import hashlib
import json

repo = Path(__file__).resolve().parents[2]
parent = Path(__file__).resolve().parent
result_path = repo / '.porting-state/portable_julia/runs/20261019T221343489132Z_leukocyte/result.json'
result = json.loads(result_path.read_text())
rows = []
for trial in result['models']['cuda']['trials']:
    artifact = Path(trial['output_artifact']['saved_path'])
    values = array.array('i')
    values.frombytes(artifact.read_bytes())
    widths = list(values[:44:2])
    assert all(v == 41 for v in values[1:44:2])
    offset = 44
    mismatches = {}
    for width in widths:
        bad = []
        for position in range(41 * width):
            actual, control = values[offset:offset+2]
            offset += 2
            assert control == position // width
            if actual != control:
                bad.append({'position': position, 'actual_row': actual,
                            'correct_row': control})
        if bad:
            mismatches[str(width)] = bad
    assert offset == len(values)
    guards_path = repo / ('.porting-state/portable_julia/leukocyte_row_probe/trial'
                          + str(trial['trial']) + '.guards.u8')
    guards = guards_path.read_bytes()
    assert len(guards) == 22 * 3 * 256 and all(v == 165 for v in guards)
    rows.append({'trial': trial['trial'], 'wrong_rows_by_width': mismatches,
                 'artifact': str(artifact.relative_to(repo)),
                 'artifact_sha256': hashlib.sha256(artifact.read_bytes()).hexdigest(),
                 'actual_guard_bytes': len(guards),
                 'guard_sha256': hashlib.sha256(guards).hexdigest()})
receipt = {
    'result': str(result_path.relative_to(repo)),
    'result_sha256': hashlib.sha256(result_path.read_bytes()).hexdigest(),
    'status': 'source_coordinate_property_failure_observed',
    'scope': 'Actual CUDA/V100 extracted original source expression, not full operator output.',
    'integer_control_pass': True,
    'wrong_rows_per_trial': 44,
    'checked_positions_per_trial': 58794,
    'full_original_operator_wrong_outputs_observed': False,
    'trials': rows,
    'measurement': 'Diagnostic only. Serialized time0 is an unmeasured schema placeholder, never benchmark performance.',
}
(parent / 'row_probe_audit.json').write_text(json.dumps(receipt, indent=2) + '\n')
header = parent / 'corrected_native/kernel_IMGVF.h'
archive = parent / 'reference_history/corrected_native_three_inherited_repairs'
archive.mkdir(parents=True, exist_ok=True)
import shutil
for name in ('detect_main.c', 'find_ellipse.c', 'kernel_IMGVF.h'):
    shutil.copy2(parent / 'corrected_native' / name, archive / name)
source = header.read_text()
old = 'i = (thread_id + offset) * one_nth;'
assert source.count(old) == 1
header.write_text(source.replace(old, '// Exact row-major coordinate; FP32 reciprocal truncation fails at n55/61.\n      i = (thread_id + offset) / n;'))
provenance_path = parent / 'reference_provenance.json'
provenance = json.loads(provenance_path.read_text())
provenance['subtypes'].append('FP32_reciprocal_row_index_rounding_to_previous_row')
provenance['current_coordinate_witness'] = 'src/leukocyte-julia/row_probe_audit.json'
provenance['minimal_corrected_before_instrumentation'] = {
    name: {'path': str((parent / 'corrected_native' / name).relative_to(repo)),
           'sha256': hashlib.sha256((parent / 'corrected_native' / name).read_bytes()).hexdigest()}
    for name in ('detect_main.c', 'find_ellipse.c', 'kernel_IMGVF.h')
}
provenance['actual_math_changes'] += ' Extra independently justified row-index correction uses source integer row-major geometry instead of FP32 reciprocal/truncation; full one-step operator checks still required.'
provenance['current_GPU_diagnostic_processes'] = 1
provenance['current_primary_GPU_processes'] = 0
provenance_path.write_text(json.dumps(provenance, indent=2) + '\n')
event = {'benchmark_id': 'leukocyte', 'implementation_model': 'shared',
         'origin_model': 'cuda', 'phase': 'native_coordinate_algorithm_validation',
         'taxonomy_code': 'S1', 'failure': True, 'is_failure': True,
         'canonical_observation_id': 'leukocyte_FP32_reciprocal_row_index_rounding',
         'detail': 'Actual V100 extracted-source expression has44wrongrows/trial: n55 seven, n61 thirtyseven. Integer control exact. Five diagnostic trials one process, not five program attempts. No full original operator wrong-output claim yet.',
         'run_id': result['run_id'], 'evidence': 'src/leukocyte-julia/row_probe_audit.json',
         'framework_gpu_attempts': 0, 'native_diagnostic_GPU_processes': 1}
with (repo / '.porting-state/portable_julia/failure_events_300/leukocyte.jsonl').open('a') as output:
    output.write(json.dumps(event) + '\n')
print({'row_probe_audit': 'pass', 'native_integer_index_secondary_written': True})
