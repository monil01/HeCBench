#!/usr/bin/env python3
"""Compare every SAIF net's numeric activity outside the GPU lock."""
import re
import sys
from pathlib import Path


def read(path):
    text = Path(path).read_text()
    duration = int(re.search(r"\(DURATION (\d+)\)", text)[1])
    rows = re.findall(
        r"\(([^\n]+)\n\(T0 (\d+)\) \(T1 (\d+)\) \(TX (\d+)\)\n"
        r"\(TC\s+(\d+)\) \(IG\s+(\d+)\s*\)\n\)", text
    )
    records = sorted((name, tuple(map(int, values))) for name, *values in rows)
    if not records:
        raise ValueError("No SAIF numeric records")
    for name, values in records:
        if sum(values[:3]) != duration:
            raise ValueError(f"Invalid duration for {name}: {values}")
    return duration, records


if len(sys.argv) != 3:
    raise SystemExit("Usage: verify.py native.saif iris.saif")
native = read(sys.argv[1])
iris = read(sys.argv[2])
if native != iris:
    mismatches = [(a, b) for a, b in zip(native[1], iris[1]) if a != b]
    print("FAIL", "native nets", len(native[1]), "IRIS nets", len(iris[1]))
    for pair in mismatches[:5]:
        print(pair)
    raise SystemExit(1)
print(f"PASS {len(native[1])} SAIF nets, exact T0/T1/TX/TC/IG, duration={native[0]}")
