#!/usr/bin/env python3
"""Reproduce the named Gaussian Float32 fixture; default checks existing bytes."""
import argparse
import hashlib
import math
from pathlib import Path
import struct


def fixture_bytes():
    n = 128
    values = [10 * math.exp(-0.01 * abs(i - j)) for i in range(n) for j in range(n)]
    return struct.pack('<' + str(n * n) + 'f', *values)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='write to a new file, refusing overwrite')
    args = parser.parse_args()
    generated = fixture_bytes()
    if args.output:
        with args.output.open('xb') as destination:
            destination.write(generated)
    else:
        saved = Path(__file__).with_name('study_input_f32.bin').read_bytes()
        if generated != saved:
            raise SystemExit('Fixture differs: this Python/libm environment does not reproduce saved bytes')
    print('sha256=' + hashlib.sha256(generated).hexdigest())
    print('bytes=' + str(len(generated)))


if __name__ == '__main__':
    main()
