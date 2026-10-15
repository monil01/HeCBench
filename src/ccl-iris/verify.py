#!/usr/bin/env python3
"""Check complete two-rank native/IRIS RCCL run logs, not timing lines."""
import pathlib
import re
import sys

for filename in sys.argv[1:]:
    text = pathlib.Path(filename).read_text()
    assert not re.search(r"\bFAIL\b|\[E\]|failed|Command exited|timed out", text), filename
    ranks = re.findall(r"MPI [Rr]ank (\d+): PASS", text)
    assert sorted(ranks) == ["0", "1"], (filename, ranks)
    if "outputs checked" in text:
        counts = re.findall(r"all (\d+) float outputs checked", text)
        assert len(counts) == 2 and all(int(n) == 1048576 for n in counts), filename
    print(f"PASS {filename}: both ranks completed with full-array checks")
assert len(sys.argv) > 1, "Provide native and IRIS run logs"
