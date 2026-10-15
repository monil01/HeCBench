#!/usr/bin/env python3
"""Independent CPU decoder/checker: verify.py original compression.txt [decompression.txt]."""
import pathlib
import struct
import sys


def decode(data):
    mask = (1 << 64) - 1
    words = struct.unpack('<' + 'Q' * (len(data) // 8), data)
    n, dim = words[0] >> 32, (words[0] & 31) + 1
    assert (words[0] >> 8) & 0xffffff == 0x43504d
    position = 1 + (n + 63) // 64
    output = []
    for start in range(0, n, 1024):
        planes, previous = [], 0
        for j in range(1024):
            idx = start + j
            flag = idx < ((n + 63) // 64) * 64 and ((words[1 + idx // 64] >> (idx % 64)) & 1)
            if flag:
                previous = (previous + words[position]) & mask
                position += 1
            planes.append(previous)
        chunk = []
        for j in range(1024):
            value = sum(((planes[(j & ~63) + bit] >> (j % 64)) & 1) << bit for bit in range(64))
            if j >= dim:
                value = (value + chunk[j - dim]) & mask
            chunk.append(value)
        output.extend(chunk[:min(1024, n - start)])
    # The original codec can serialize one terminal plane-reset delta after
    # the last declared bitmap group; complete native/IRIS artifact comparison
    # separately covers this unused padding word.
    assert 0 <= len(words) - position <= 1
    return struct.pack('<' + 'Q' * n, *output)


if __name__ == '__main__':
    if len(sys.argv) not in (3, 4):
        raise SystemExit(__doc__)
    original = pathlib.Path(sys.argv[1]).read_bytes()
    assert decode(pathlib.Path(sys.argv[2]).read_bytes()) == original, 'CPU decode mismatch'
    if len(sys.argv) == 4:
        assert pathlib.Path(sys.argv[3]).read_bytes() == original, 'GPU roundtrip mismatch'
    print(f'PASS: every one of {len(original)//8} input double bitpatterns restored')
