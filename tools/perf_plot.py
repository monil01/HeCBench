#!/usr/bin/env python3
"""Plot the perf sweep. One PDF, embeddable in the paper.

Two side-by-side subplots at the medium input size:
  (a) absolute kernel time (µs, log-y)
  (b) slowdown relative to CUDA (linear-y, capped)

Both use the same color-per-model palette so the reader can align them.
"""
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
               "Hecbench-agent/IEEEtran/fig_perf.pdf")

BENCHES = ["accuracy", "adam", "adjacent", "atan2", "aidw"]
MODELS  = ["cuda", "omp", "serial", "triton", "julia", "rust"]
LABELS  = {"cuda":"CUDA","omp":"OMP-tgt","serial":"Serial",
           "triton":"Triton","julia":"Julia","rust":"Rust"}
# Colorblind-safe palette from Wong 2011
COLORS  = {"cuda":"#0072B2",   # blue
           "omp":"#E69F00",    # orange
           "serial":"#999999", # grey
           "triton":"#009E73", # green
           "julia":"#CC79A7",  # magenta
           "rust":"#D55E00"}   # vermillion

def load(size="medium"):
    data = {}
    with open(CSV) as f:
        for r in csv.DictReader(f):
            if r["size"] != size: continue
            v = r["exec_time_us"]
            if not v:  # timeout / no time line
                continue
            data[(r["bench"], r["model"])] = float(v)
    return data

def main():
    med = load("medium")
    lg  = load("large")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.3), sharey=False)

    # ---- Absolute (log y) ------------------------------------------------
    n_b, n_m = len(BENCHES), len(MODELS)
    x = np.arange(n_b)
    w = 0.13
    for i, m in enumerate(MODELS):
        heights = [med.get((b, m), 0) for b in BENCHES]
        # replace 0 (missing) with small so it shows a stub
        heights_plot = [h if h > 0 else np.nan for h in heights]
        ax1.bar(x + (i - n_m/2 + 0.5) * w, heights_plot, width=w,
                color=COLORS[m], label=LABELS[m], edgecolor='black',
                linewidth=0.3)
    ax1.set_yscale("log")
    ax1.set_xticks(x)
    ax1.set_xticklabels(BENCHES, rotation=0, ha="center", fontsize=9)
    ax1.set_ylabel("Kernel time (µs)  —  log scale", fontsize=9)
    ax1.set_title("(a) Absolute kernel time, medium input", fontsize=10)
    ax1.grid(True, axis="y", which="both", ls=":", lw=0.5, alpha=0.4)
    ax1.set_axisbelow(True)

    # ---- Slowdown vs CUDA (log y) ---------------------------------------
    for i, m in enumerate(MODELS):
        if m == "cuda": continue
        ratios = []
        for b in BENCHES:
            c = med.get((b, "cuda"))
            v = med.get((b, m))
            ratios.append(v / c if (c and v) else np.nan)
        ax2.bar(x + (i - n_m/2 + 0.5) * w, ratios, width=w,
                color=COLORS[m], label=LABELS[m], edgecolor='black',
                linewidth=0.3)
    ax2.axhline(1.0, color="#0072B2", ls="--", lw=1, alpha=0.7)
    ax2.set_yscale("log")
    ax2.set_xticks(x)
    ax2.set_xticklabels(BENCHES, rotation=0, ha="center", fontsize=9)
    ax2.set_ylabel("Slowdown vs CUDA  ($\\times$, log)", fontsize=9)
    ax2.set_title("(b) Slowdown relative to CUDA, medium input", fontsize=10)
    ax2.grid(True, axis="y", which="both", ls=":", lw=0.5, alpha=0.4)
    ax2.set_axisbelow(True)

    # Shared legend at bottom
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6,
               bbox_to_anchor=(0.5, -0.04), frameon=False, fontsize=9)

    fig.tight_layout()
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF, bbox_inches="tight")
    print(f"wrote {OUT_PDF}")

if __name__ == "__main__":
    main()
