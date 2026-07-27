#!/usr/bin/env python3
"""Scaling plot: one panel per benchmark showing exec time vs input size,
one line per programming model."""
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CSV = Path("/tmp/claude-10384/-home-imo-HeCBench/"
           "471e31c0-3c23-4cc0-bd9c-b68cd31894f3/scratchpad/perf.csv")
OUT_PDF = Path("/tmp/claude-10384/-home-imo-HeCBench/"
               "471e31c0-3c23-4cc0-bd9c-b68cd31894f3/scratchpad/"
               "Hecbench-agent/IEEEtran/fig_scaling.pdf")

BENCHES = ["accuracy", "adam", "adjacent", "atan2", "aidw"]
SIZES   = ["small", "medium", "large"]
MODELS  = ["cuda", "omp", "serial", "triton", "julia", "rust"]
LABELS  = {"cuda":"CUDA","omp":"OMP-tgt","serial":"Serial",
           "triton":"Triton","julia":"Julia","rust":"Rust"}
COLORS  = {"cuda":"#0072B2","omp":"#E69F00","serial":"#999999",
           "triton":"#009E73","julia":"#CC79A7","rust":"#D55E00"}
MARKERS = {"cuda":"o","omp":"s","serial":"^","triton":"D","julia":"v","rust":"P"}

def load():
    data = {}   # (bench, model, size) -> (first_arg, exec_us)
    with open(CSV) as f:
        for r in csv.DictReader(f):
            v = r["exec_time_us"]
            if not v: continue
            first_arg = float(r["first_arg"]) if r["first_arg"] else 0
            data[(r["bench"], r["model"], r["size"])] = (first_arg, float(v))
    return data

def main():
    data = load()

    fig, axes = plt.subplots(1, 5, figsize=(11, 2.6), sharey=False)

    for ax, bench in zip(axes, BENCHES):
        # x-axis: first_arg values in size order small/medium/large
        xs = []
        for sz in SIZES:
            for m in MODELS:
                if (bench, m, sz) in data:
                    xs.append(data[(bench, m, sz)][0]); break
        xs = sorted(set(xs))
        if len(xs) < 2:
            continue

        for m in MODELS:
            ys = []
            for x in xs:
                # find matching size for this first_arg
                match = None
                for sz in SIZES:
                    key = (bench, m, sz)
                    if key in data and data[key][0] == x:
                        match = data[key][1]; break
                ys.append(match)
            # plot only points that exist
            ax.plot(xs, ys, marker=MARKERS[m], color=COLORS[m],
                    label=LABELS[m], linewidth=1.2, markersize=4)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(bench, fontsize=10)
        ax.set_xlabel("input size (first arg)", fontsize=8)
        ax.grid(True, which="both", ls=":", lw=0.4, alpha=0.4)
        ax.tick_params(labelsize=7)

    axes[0].set_ylabel("kernel time (µs)", fontsize=9)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6,
               bbox_to_anchor=(0.5, -0.06), frameon=False, fontsize=9)

    fig.tight_layout()
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF, bbox_inches="tight")
    print(f"wrote {OUT_PDF}")

if __name__ == "__main__":
    main()
