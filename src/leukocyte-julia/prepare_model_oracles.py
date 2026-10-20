#!/usr/bin/env python3
from pathlib import Path
import array
import hashlib
import json
import mmap
import struct
import argparse
import re
parser=argparse.ArgumentParser();parser.add_argument("--model",required=True);parser.add_argument("--raw-root",required=True);parser.add_argument("--prefix",required=True);parser.add_argument("--process-id",type=int);options=parser.parse_args()

parent = Path(__file__).resolve().parent
repo = parent.parents[1]
raw_root = Path(options.raw_root)
paths = []
for candidate in raw_root.glob(options.model+'_*trial1_case100.timings.jsonl'):
    if re.match(re.escape(options.model)+r'_\d+_',candidate.name) and (options.process_id is None or candidate.name.startswith(f'{options.model}_{options.process_id}_')) and len(candidate.read_text().splitlines()) == 102:
        paths.append(candidate)
assert len(paths) == 1
stem = str(paths[0]).removesuffix('case100.timings.jsonl')
seen_shapes = set()
jobs = []
initial = None
summary = []
for frames in (1, 3, 100):
    base = Path(stem + 'case' + str(frames))
    layout = [json.loads(line) for line in base.with_suffix('.layout.jsonl').read_text().splitlines()]
    timings = [json.loads(line) for line in base.with_suffix('.timings.jsonl').read_text().splitlines()]
    timing_by_frame = {entry['frame']: entry for entry in timings if entry['stage'] == 'MGVF'}
    file = base.with_suffix('.raw.bin').open('rb')
    memory = mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ)
    guards = base.with_suffix('.guards.u8').read_bytes()
    assert all(value == 165 for value in guards)
    before = {}
    for entry in layout:
        if entry['kind'] != 'gpu_buffers':
            continue
        fields = {}
        for buffer in entry['buffers']:
            offset = buffer['raw_offset_bytes'] + buffer['payload_offset_bytes']
            count = buffer['payload_bytes']
            actual = memory[offset:offset+count]
            dtype = 'i' if buffer['dtype'] == 'int32' else 'f'
            values = array.array(dtype)
            values.frombytes(actual)
            fields[buffer['name']] = values
            guard_offset = buffer['guard_offset_bytes']
            expected = memory[offset-256:offset] + memory[offset+count:offset+count+256]
            assert guards[guard_offset:guard_offset+512] == expected
        key = (entry['stage'], entry['frame'])
        if entry['phase'] == 'before':
            before[key] = fields
            continue
        inputs = before[key]
        if entry['stage'] == 'GICOV' and initial is None:
            initial = {'m': 219, 'n': 640, 'inputs': inputs, 'gicov': fields['d_gicov']}
        if entry['stage'] == 'DILATION' and 'dilation' not in initial:
            initial['strel'] = inputs['d_strel']
            initial['dilation'] = fields['d_img_dilated']
        if entry['stage'] == 'MGVF':
            offsets, ms, ns = (inputs[name] for name in ('d_I_offsets', 'd_m_array', 'd_n_array'))
            image, start = inputs['d_I_all'], inputs['d_IMGVF_all']
            assert image == start
            assert len(offsets) == len(ms) == len(ns) == len(fields['d_iterations'])
            assert sum(m*n for m, n in zip(ms, ns)) == len(image)
            expected_offset = 0
            for cell, (offset, m, n, iterations) in enumerate(zip(offsets, ms, ns, fields['d_iterations'])):
                assert offset == expected_offset
                assert 1 <= iterations <= 500
                expected_offset += m*n
                assert timing_by_frame[entry['frame']]['iterations'][cell] == iterations
                shape = (m, n)
                if shape not in seen_shapes:
                    seen_shapes.add(shape)
                    jobs.append((frames, entry['frame'], cell, m, n, iterations,
                                 image[offset:offset+m*n], fields['d_IMGVF_all'][offset:offset+m*n]))
    summary.append({'frames': frames, 'raw_bytes': len(memory), 'actual_guard_bytes': len(guards),
                    'raw_sha256': hashlib.sha256(memory).hexdigest(),
                    'guard_sha256': hashlib.sha256(guards).hexdigest(),
                    'GPU_launches': len(timings),
                    'cell_iteration_sum': sum(sum(t['iterations']) for t in timings)})
    memory.close()
    file.close()
assert initial is not None
with (parent / (options.prefix+'initial_detector_oracle_input.bin')).open('wb') as output:
    output.write(struct.pack('<6i', 219, 640, 7, 150, 25, 25))
    for name in ('d_grad_x', 'd_grad_y', 'd_cos_angle', 'd_sin_angle', 'd_tX', 'd_tY'):
        output.write(initial['inputs'][name].tobytes())
    for name in ('strel', 'gicov', 'dilation'):
        output.write(initial[name].tobytes())
with (parent / (options.prefix+'selected_flow_oracle_input.bin')).open('wb') as output:
    output.write(struct.pack('<i', len(jobs)))
    for frames, frame, cell, m, n, iterations, data, actual in jobs:
        output.write(struct.pack('<6i', frames, frame, cell, m, n, iterations))
        output.write(data.tobytes())
        output.write(actual.tobytes())
metadata = {'pilot_case_streams': summary,
            'source_scope': 'Actual '+options.model+' first checked trial, first actual patch per distinct(m,n), as predeclared; detector and flow outputs from same model raw streams.',
            'selected_jobs': [{'frames': j[0], 'frame': j[1], 'cell': j[2], 'm': j[3], 'n': j[4], 'iterations': j[5]} for j in jobs],
            'assets': [{'path': name, 'bytes': (parent/name).stat().st_size,
                        'sha256': hashlib.sha256((parent/name).read_bytes()).hexdigest()}
                       for name in ((options.prefix+'initial_detector_oracle_input.bin'), (options.prefix+'selected_flow_oracle_input.bin'))]}
(parent / (options.prefix+'full_oracle_input_manifest.json')).write_text(json.dumps(metadata, indent=2) + '\n')
print({'case_streams': summary, 'selected_shapes': len(jobs), 'assets': metadata['assets']})
