#!/usr/bin/env python3
"""Convert scratchpad/perf.csv into a LaTeX tabular for the paper.

Reads the CSV produced by tools/perf_sweep.py and prints rows suitable for
insertion into the tabular in 4.perfomrance_comparison.tex.
"""
import csv, sys
from collections import defaultdict
from pathlib import Path

CSV = Path("/tmp/claude-10384/-home-imo-HeCBench/"
           "471e31c0-3c23-4cc0-bd9c-b68cd31894f3/scratchpad/perf.csv")
OUT_TEX = Path("/tmp/claude-10384/-home-imo-HeCBench/"
               "471e31c0-3c23-4cc0-bd9c-b68cd31894f3/scratchpad/perf_rows.tex")

BENCHES = ["accuracy","adam","adjacent","atan2","aidw"]
SIZES   = ["small","medium","large"]
MODELS  = ["cuda","omp","serial","triton","julia","rust"]

def fmt(us: str) -> str:
    if not us:
        return "--"
    v = float(us)
    if v < 100:
        return f"{v:.2f}"
    if v < 1e4:
        return f"{v:.1f}"
    if v < 1e6:
        return f"{v/1e3:.1f}k"
    return f"{v/1e6:.2f}M"

def main():
    if not CSV.exists():
        print(f"perf.csv not found at {CSV}"); return 1

    # Load rows
    data = defaultdict(dict)  # (bench,size,model) -> exec_time_us
    firstarg = {}             # (bench,size) -> first_arg
    with open(CSV) as f:
        for r in csv.DictReader(f):
            k = (r["bench"], r["size"], r["model"])
            data[k] = r["exec_time_us"]
            firstarg[(r["bench"], r["size"])] = r["first_arg"]

    lines = []
    for bench in BENCHES:
        for i, size in enumerate(SIZES):
            first = firstarg.get((bench, size), "")
            label = size if i == 0 else size
            row_parts = []
            if i == 0:
                row_parts.append(f"\\texttt{{{bench}}}")
            else:
                row_parts.append("")
            row_parts.append(f"{label}({first})")
            for m in MODELS:
                us = data.get((bench, size, m), "")
                row_parts.append(fmt(us))
            lines.append(" & ".join(row_parts) + " \\\\")
        lines.append("\\midrule")

    OUT_TEX.write_text("\n".join(lines) + "\n")
    print(f"Wrote {OUT_TEX}")
    print("---sample---")
    for ln in lines[:15]:
        print(ln)

if __name__ == "__main__":
    sys.exit(main())
