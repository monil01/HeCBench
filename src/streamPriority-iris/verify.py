#!/usr/bin/env python3
"""Whole-output native/IRIS comparison and actual stream-priority evidence."""
import argparse
import pathlib
import re
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--elements", type=int, required=True)
parser.add_argument("--native-prefix", required=True)
parser.add_argument("--iris-prefix", required=True)
parser.add_argument("--native-log", required=True)
parser.add_argument("--iris-log", required=True)
args = parser.parse_args()
native_log = pathlib.Path(args.native_log).read_text()
iris_log = pathlib.Path(args.iris_log).read_text()
for text in (native_log, iris_log):
    assert not re.search(r"\bFAIL\b|\[E\]|Command exited|Command terminated|timed out|Segmentation|Aborted|\bERROR\b|stream error", text)
assert len(re.findall(r"^PASS$", native_log, re.M)) == 2
assert len(re.findall(r"streams: .*; PASS; all", iris_log)) == 2
match = re.search(r"train=(-?\d+) prefetch=(-?\d+); supported range=(-?\d+)\.\.(-?\d+)", iris_log)
assert match, "Missing real HIP stream priority queries"
high, low, greatest, least = map(int, match.groups())
assert high == greatest and low == least and high < low, "Distinct priorities not demonstrated"
for mode in ("baseline", "priority"):
    native = np.fromfile(args.native_prefix + "." + mode + ".bin", dtype=np.float32)
    iris = np.fromfile(args.iris_prefix + "." + mode + ".bin", dtype=np.float32)
    assert native.size == iris.size == args.elements * 4
    assert np.isfinite(native).all() and np.isfinite(iris).all()
    assert native.tobytes() == iris.tobytes(), mode + " is not byte-exact"
    print(f"PASS {mode}: all {native.size} act/grad/weights/prefetch floats byte-exact native; high={high}, low={low}")
