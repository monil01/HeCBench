#!/usr/bin/env python3
from pathlib import Path
import json
import os
import subprocess

parent = Path(__file__).resolve().parent
original = parent / 'original_native'
logs = parent / 'readiness_logs'
program = parent / 'study_original_host_asan'
command = [
    'g++', '-O1', '-g', '-std=c++17', '-fsanitize=address',
    '-fno-omit-frame-pointer',
    '-I' + str(original / 'meschach_lib'),
    '-I/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda/12.9/include',
    str(parent / 'study_host_probe.cpp'), '-x', 'c++',
    str(original / 'find_ellipse.c'), str(original / 'misc_math.c'),
    str(original / 'avilib.c'), '-x', 'none',
    str(parent / 'corrected_native/meschach_lib/meschach.a'),
    '-lm', '-o', str(program),
]
with (logs / 'original_interp_ASan_build.log').open('w') as output:
    build = subprocess.run(command, stdout=output, stderr=subprocess.STDOUT)
results = {'build_command': command, 'build_returncode': build.returncode,
           'GPU_attempts': 0, 'cases': []}
if build.returncode == 0:
    for mode in ('affine', 'invalid'):
        log = logs / ('original_interp_ASan_' + mode + '.log')
        with log.open('w') as output:
            run = subprocess.run([str(program), mode], stdout=output,
                                 stderr=subprocess.STDOUT,
                                 env={**os.environ, 'ASAN_OPTIONS': 'detect_leaks=0'},
                                 timeout=30)
        results['cases'].append({'mode': mode, 'returncode': run.returncode,
                                 'log': str(log),
                                 'observed_ASan_heap_buffer_overflow':
                                 'heap-buffer-overflow' in log.read_text()})
results['interpretation'] = (
    'Deliberately invalid y2.25 in3x4 exercises missing original interpolation '
    'domain validation; does not establish a current full-CLI crash/cause. '
    'Valid affine case should pass. Corrected guard rejects invalid coordinates '
    'without clipping/formula change.'
)
(logs / 'original_interp_ASan_results.json').write_text(json.dumps(results, indent=2) + '\n')
print(results)
