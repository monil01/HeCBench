#!/usr/bin/env python3
"""Validate saved numerical and global GPU overlap diagnostics."""
import json
import pathlib
import re
import sys

results = []
for filename in sys.argv[1:]:
    text = pathlib.Path(filename).read_text()
    assert re.search(r"^PASS$", text, re.M), filename
    assert not re.search(r"\[E\]|\bFAIL\b|Command exited|timed out", text), filename
    match = re.search(r"sum=(\d+) expected=(\d+); all (\d+)", text)
    if not match:
        assert "Starting..." in text and "time clocks =" in text, filename
        results.append({"file": filename, "native_verdict": "PASS"})
        continue
    actual, expected, count = map(int, match.groups())
    clocks = int(re.search(r"time clocks = (\d+)", text)[1])
    # Exact count of residues 0,1,2 in [0, clocks), independently of GPU output.
    oracle = count * (3 * (clocks // 3) + max(0, clocks % 3 - 1))
    assert actual == expected == oracle, filename
    intervals = [tuple(map(int, p)) for p in re.findall(r"interval\[\d+\]=(\d+),(\d+)", text)]
    overlap = None
    if intervals:
        assert len(intervals) == count and all(a < b for a, b in intervals), filename
        overlap = sum(a < d and c < b for i, (a, b) in enumerate(intervals) for c, d in intervals[:i])
        assert count == 1 or overlap > 0, filename
        assert int(re.search(r"overlapping producer pairs: (\d+)", text)[1]) == overlap
    results.append({"file": filename, "producers": count, "sum": actual, "overlapping_pairs": overlap})
assert results, "Provide the native, numerical, and overlap log paths"
print(json.dumps({"status": "ok", "checks": results}, indent=2))
