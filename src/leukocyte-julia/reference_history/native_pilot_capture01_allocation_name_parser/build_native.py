#!/usr/bin/env python3
"""Build the complete native application in the verifier's temporary directory."""
from pathlib import Path
import argparse
import json
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('--output', default='study_native')
parser.add_argument('--nvcc', default='/usr/bin/nvcc')
options = parser.parse_args()
parent = Path(__file__).resolve().parent
native = parent / 'instrumented_native'
matrix = parent / 'corrected_native/meschach_lib'
commands = []
library_command = ['make', '-C', str(matrix), 'all', 'CC=g++',
                   'CFLAGS=-O3 -std=c++17 -fPIC -x c++']
commands.append(library_command)
subprocess.run(library_command, check=True)
sources = [parent / 'study_native_driver.cu'] + [native / name for name in [
    'detect_main.c', 'track_ellipse.c', 'track_ellipse_gpu.c',
    'find_ellipse.c', 'misc_math.c', 'avilib.c',
]]
objects = []
for index, source in enumerate(sources):
    target = Path.cwd() / ('native_' + str(index) + '.o')
    command = [options.nvcc, '-O3', '-std=c++17', '-arch=sm_70',
               '-I' + str(matrix), '-x', 'cu', '-c', str(source), '-o', str(target)]
    commands.append(command)
    print('NATIVE_BUILD_COMMAND ' + json.dumps(command), flush=True)
    subprocess.run(command, check=True)
    objects.append(str(target))
command = [options.nvcc] + objects + [str(matrix / 'meschach.a'), '-lm', '-o', options.output]
commands.append(command)
print('NATIVE_BUILD_COMMAND ' + json.dumps(command), flush=True)
subprocess.run(command, check=True)
Path('native_build_commands.json').write_text(json.dumps(commands, indent=2) + '\n')
