#!/usr/bin/env python3
"""Compare full-array allocation checks from native and IRIS log files."""
import pathlib
import re
import sys

observed = []
for filename in sys.argv[1:]:
    text = pathlib.Path(filename).read_text()
    assert not re.search(r"\bFAIL\b|\[E\]|failed|Command exited|timed out", text), filename
    cases = [tuple(map(int, x)) for x in re.findall(r"PASS allocation kind=(\d+) bytes=(\d+) buffers=(\d+); every element checked", text)]
    assert cases and {c[0] for c in cases} == {0, 1, 2}, filename
    assert len(cases) == len(set(cases)), filename
    for kind in range(3):
        assert [(b, n) for k, b, n in cases if k == kind] == [(b, n) for k, b, n in cases if k == 0], filename
    observed.append(cases)
    print(f"PASS {filename}: {len(cases)} physical allocation cases, {sum(n*b//4 for _, b, n in cases)} integer outputs checked")
assert len(observed) >= 2 and all(x == observed[0] for x in observed), "Native/IRIS case coverage mismatch"
