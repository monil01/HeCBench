#!/usr/bin/env python3
"""Compare every unordered intersection against native GPU and Python sets."""
import pathlib
import re
import struct
import sys

def dataset(path):
    data = pathlib.Path(path).read_bytes()
    k, universe, total = struct.unpack_from('<IIQ', data)
    sizes = struct.unpack_from(f'<{k}I', data, 16)
    elements = struct.unpack_from(f'<{total}I', data, 16+4*k)
    assert len(data) == 16+4*k+4*total and sum(sizes) == total
    sets, off = [], 0
    for n in sizes:
        row = elements[off:off+n]
        assert all(1 <= v <= universe for v in row) and len(set(row)) == n
        sets.append(set(row)); off += n
    return sets

def result(path):
    out = {}
    for line in pathlib.Path(path).read_text().splitlines():
        m = re.fullmatch(r'\((\d+),(\d+)\): (\S+)', line)
        assert m, line
        a,b = int(m[1])-1,int(m[2])-1
        assert a < b and (a,b) not in out
        out[a,b] = float(m[3])
    return out

if __name__ == '__main__':
    sets = dataset(sys.argv[1]); native = result(sys.argv[2]); iris = result(sys.argv[3])
    reference = {(a,b): len(sets[a] & sets[b]) for a in range(len(sets)) for b in range(a+1,len(sets))}
    assert native == reference, 'Native differs from independent full set-intersection oracle'
    assert iris == reference, 'IRIS differs from independent full set-intersection oracle'
    print(f'PASS all {len(reference)} intersections exact: native HIP, IRIS DMem/hipBLAS, independent Python sets; distinct results {len(set(reference.values()))}')
