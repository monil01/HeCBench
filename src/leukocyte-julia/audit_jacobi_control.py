#!/usr/bin/env python3
"""Recheck every saved point with an independent Python scalar Jacobi oracle."""
from pathlib import Path
import array
import hashlib
import json
import math
import struct

parent = Path(__file__).resolve().parent
repo = parent.parents[1]
result_path = repo / '.porting-state/portable_julia/runs/20261019T222350397120Z_leukocyte/result.json'
result = json.loads(result_path.read_text())
f32 = lambda x: struct.unpack('f', struct.pack('f', x))[0]
inverse_pi = f32(1.0 / f32(3.14159))
diffusion = f32(.5 / 5)
shapes = [(7, 9), (41, 41), (41, 47), (41, 55), (41, 61), (41, 81), (41, 61), (41, 61)]
neighbors = [(-1, 0, -1), (1, 0, 1), (0, -1, -1), (0, 1, 1),
             (-1, 1, 0), (1, 1, 2), (-1, -1, -2), (1, -1, 0)]
trial_results = []
for trial in result['models']['cuda']['trials']:
    path = Path(trial['output_artifact']['saved_path'])
    saved = array.array('d')
    saved.frombytes(path.read_bytes())
    prefix_end = result['input_prefix_count']
    prefix_offset = 0
    output_offset = prefix_end
    cases = []
    for case_id, (expected_m, expected_n) in enumerate(shapes):
        m, n, offset, vx, vy, epsilon, cutoff, iterations = saved[prefix_offset:prefix_offset+8]
        prefix_offset += 8
        m, n = int(m), int(n)
        assert (m, n) == (expected_m, expected_n)
        assert offset == 0 and vx == vy == iterations == 1 and cutoff == 0
        data = saved[prefix_offset:prefix_offset+m*n]
        prefix_offset += m*n
        inverse_e = f32(1.0 / epsilon)
        worst_old = worst_new = worst_oracle = 0.0
        wrong_old = 0
        examples = []
        for row in range(m):
            for col in range(n):
                initial = data[row*n+col]
                weighted = 0.0
                for dr, dc, direction in neighbors:
                    r = max(0, min(m-1, row+dr))
                    c = max(0, min(n-1, col+dc))
                    delta = data[r*n+c] - initial
                    weight = math.atan(delta*direction*inverse_e)*inverse_pi + .5
                    weighted += weight*delta
                prediction = initial + diffusion*weighted
                prediction -= .2*initial*(prediction-initial)
                old, new, c_oracle = saved[output_offset:output_offset+3]
                output_offset += 3
                old_error = abs(old-prediction)
                new_error = abs(new-prediction)
                worst_old = max(worst_old, old_error)
                worst_new = max(worst_new, new_error)
                worst_oracle = max(worst_oracle, abs(c_oracle-prediction))
                assert new_error <= 5e-6
                assert abs(c_oracle-prediction) <= 1e-14
                if case_id == 7:
                    assert initial == old == new == .375
                if old_error > 5e-6:
                    wrong_old += 1
                    if len(examples) < 4:
                        examples.append({'row': row, 'column': col, 'old': old,
                                         'corrected': new, 'oracle': prediction})
        cases.append({'case': case_id, 'm': m, 'n': n,
                      'old_wrong_values': wrong_old,
                      'old_max_abs_error': worst_old,
                      'corrected_max_abs_error': worst_new,
                      'independent_CPU_oracles_max_abs_difference': worst_oracle,
                      'examples': examples})
    assert prefix_offset == prefix_end and output_offset == len(saved)
    guards_path = repo / '.porting-state/portable_julia/leukocyte_jacobi_control' / ('trial' + str(trial['trial']) + '.guards.u8')
    guards = guards_path.read_bytes()
    assert len(guards) == 12288 and all(x == 165 for x in guards)
    trial_results.append({'trial': trial['trial'], 'cases': cases,
                          'artifact_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                          'actual_guard_bytes': len(guards),
                          'guards_sha256': hashlib.sha256(guards).hexdigest()})
receipt = {
    'result': str(result_path.relative_to(repo)),
    'result_sha256': hashlib.sha256(result_path.read_bytes()).hexdigest(),
    'status': 'positive_corrected_operator_and_negative_old_coordinate_control_pass',
    'checked_points': 5*16750,
    'corrected_max_abs_error': max(c['corrected_max_abs_error'] for t in trial_results for c in t['cases']),
    'old_max_abs_error': max(c['old_max_abs_error'] for t in trial_results for c in t['cases']),
    'old_wrong_values_per_trial': [sum(c['old_wrong_values'] for c in t['cases']) for t in trial_results],
    'actual_guard_bytes': sum(t['actual_guard_bytes'] for t in trial_results),
    'trials': trial_results,
    'scope': 'One-step controls, including actual source epsilon and extra smooth-Heaviside/constant controls. Old header already has the prior Jacobi barrier; only row arithmetic differs. Not whole-video or500-step accuracy/timing evidence.',
    'tolerance_interpretation': '5e-6 is a predeclared local validation tolerance motivated by FP32 operation scale. It is not a proved universal atanf/500-step forward-error bound. Report actual saved-point error separately.',
}
(parent / 'jacobi_control_audit.json').write_text(json.dumps(receipt, indent=2) + '\n')
print({k: v for k, v in receipt.items() if k != 'trials'})
