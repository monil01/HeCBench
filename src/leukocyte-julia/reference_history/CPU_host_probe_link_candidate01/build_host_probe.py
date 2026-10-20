#!/usr/bin/env python3
from pathlib import Path
import json
import subprocess
import time

parent = Path(__file__).resolve().parent
native = parent / 'corrected_native'
logs = parent / 'readiness_logs'
command = [
    'g++', '-O3', '-std=c++17',
    '-I' + str(native / 'meschach_lib'),
    '-I/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda/12.9/include',
    str(parent / 'study_host_probe.cpp'), '-x', 'c++',
    str(native / 'find_ellipse.c'), str(native / 'misc_math.c'),
    str(native / 'avilib.c'), '-x', 'none',
    str(native / 'meschach_lib/meschach.a'), '-lm',
    '-o', str(parent / 'study_host_probe'),
]
started = time.monotonic()
with (logs / 'host_probe_build_01.log').open('w') as output:
    build = subprocess.run(command, stdout=output, stderr=subprocess.STDOUT)
metadata = {'command': command, 'returncode': build.returncode,
            'seconds': time.monotonic() - started,
            'phase': 'CPU_reference_scalar_matrix_interpolation', 'GPU_attempts': 0}
(logs / 'host_probe_build_01.json').write_text(json.dumps(metadata, indent=2) + '\n')
print(metadata)
print((logs / 'host_probe_build_01.log').read_text()[-1800:])
if build.returncode == 0:
    results = []
    for name, arguments in [
        ('matrix', [str(parent / 'TMatrix_reference.f64')]),
        ('affine', ['affine']),
        ('invalid', ['invalid']),
    ]:
        log = logs / ('host_probe_' + name + '.log')
        with log.open('w') as output:
            result = subprocess.run([str(parent / 'study_host_probe')] + arguments,
                                    stdout=output, stderr=subprocess.STDOUT)
        results.append({'mode': name, 'returncode': result.returncode,
                        'expected': 1 if name == 'invalid' else 0, 'log': str(log)})
    (logs / 'host_probe_results.json').write_text(json.dumps(results, indent=2) + '\n')
    print(results)
