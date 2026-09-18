#!/usr/bin/env python3
"""Cross-model performance sweep across HeCBench benchmarks.

For each (benchmark, model, input_size) triple:
  - build if needed
  - run with a per-cell timeout
  - parse the "Average execution time" line
  - append to a CSV

Output columns: bench, model, size_label, first_arg, wall_seconds,
                exec_time_us, exit_code, notes.

Only runs benchmarks/models we know work on this box. Skips HIP, SYCL, Mojo.
"""
from __future__ import annotations
import argparse, csv, os, re, subprocess, sys, time
from pathlib import Path

REPO = Path("/home/imo/HeCBench")
SRC  = REPO / "src"
STATE = Path(os.environ.get("STATE", REPO / ".porting-state"))
OUT_CSV = STATE / "perf.csv"

HPC_SDK = "/opt/nvidia/hpc_sdk/Linux_x86_64/26.3"
CUDA_ARCH = "sm_120"
NVC_SM    = "cc120"

# Build env
ENV = os.environ.copy()
ENV["PATH"] = (f"{HPC_SDK}/compilers/bin:{HPC_SDK}/cuda/bin:"
               f"{os.path.expanduser('~')}/.cargo/bin:"
               f"{os.path.expanduser('~')}/.juliaup/bin:"
               + ENV.get("PATH", ""))
ENV["LD_LIBRARY_PATH"] = (
    f"{HPC_SDK}/compilers/lib:{HPC_SDK}/cuda/lib64:"
    f"{HPC_SDK}/cuda/13.1/targets/x86_64-linux/lib:"
    + ENV.get("LD_LIBRARY_PATH", ""))

# -------------------------------------------------------------------------
# Benchmark definitions
# -------------------------------------------------------------------------
# Each entry: bench -> list of (size_label, args_list)
# We pick benchmarks that scale cleanly with the first argument.

BENCHMARKS = {
    "accuracy": [
        ("small",  ["2048", "10000", "10", "50"]),
        ("medium", ["8192", "10000", "10", "50"]),
        ("large",  ["16384", "10000", "10", "50"]),
    ],
    "adam": [
        ("small",  ["10000", "50", "50"]),
        ("medium", ["100000", "50", "50"]),
        ("large",  ["1000000", "50", "50"]),
    ],
    "adjacent": [
        ("small",  ["100000", "100"]),
        ("medium", ["1000000", "100"]),
        ("large",  ["10000000", "100"]),
    ],
    "atan2": [
        ("small",  ["100000", "100"]),
        ("medium", ["1000000", "100"]),
        ("large",  ["10000000", "100"]),
    ],
    "aidw": [
        ("small",  ["1", "1", "50"]),
        ("medium", ["10", "1", "50"]),
        ("large",  ["50", "1", "50"]),
    ],
}

MODELS = ["cuda", "omp", "serial", "triton", "julia", "rust"]

# Per-model, per-size timeouts (seconds). Serial can be very slow on large.
def timeout_for(model: str, size: str) -> int:
    if model == "serial":
        return {"small": 60, "medium": 180, "large": 480}[size]
    if model in ("triton", "julia"):
        return {"small": 60, "medium": 120, "large": 240}[size]
    return {"small": 30, "medium": 60, "large": 180}[size]

# -------------------------------------------------------------------------
# Build helpers
# -------------------------------------------------------------------------
def build_cmd(model: str, path: Path):
    if model == "cuda":
        return [["make", "clean"], ["make", f"ARCH={CUDA_ARCH}"]]
    if model == "omp":
        if (path / "Makefile.nvc").exists():
            return [["make", "-f", "Makefile.nvc", "clean"],
                    ["make", "-f", "Makefile.nvc", f"SM={NVC_SM}"]]
        return [["make", "clean"], ["make"]]
    if model == "serial":
        return [["make", "clean"], ["make"]]
    if model == "rust":
        return [["cargo", "build", "--release", "--quiet"]]
    return None  # triton, julia don't need pre-build

def run_binary_cmd(model: str, path: Path, args: list[str]):
    if model in ("cuda", "omp", "serial"):
        return [str(path / "main"), *args]
    if model == "rust":
        # Discover binary name in target/release
        d = path / "target" / "release"
        if d.is_dir():
            for f in d.iterdir():
                if f.is_file() and os.access(f, os.X_OK) and not f.suffix and f.stat().st_size > 100000:
                    return [str(f), *args]
        return None
    if model == "triton":
        return ["/noback/imo/miniconda3/bin/python3", "main.py", *args]
    if model == "julia":
        return ["julia", "--project=/home/imo/HeCBench/src/_julia_env",
                "main.jl", *args]
    return None

# -------------------------------------------------------------------------
# Parse "Average execution time" from stdout
# -------------------------------------------------------------------------
# Common patterns:
#   Average execution time of accuracy kernel: 208.375100 (us)
#   Average kernel execution time 16.772503 (ms)
#   Average execution time of the kernels (thread block size = 1024): 11.5 (us)
#   Average execution time: 197.6 (us)
#   Average execution time of AIDW_Kernel  0.00181... (s)
_TIME_LINE_RE = re.compile(
    r"(?:average|elapsed)[^\n]*?([-+]?\d+\.?\d*(?:e[-+]?\d+)?)\s*\((us|ms|s)\)",
    re.I)

def parse_time_us(stdout: str) -> float | None:
    """Return the last reported average execution time in microseconds."""
    hits = _TIME_LINE_RE.findall(stdout)
    if not hits:
        return None
    val, unit = hits[-1]
    v = float(val)
    if unit.lower() == "ms":
        v *= 1000.0
    elif unit.lower() == "s":
        v *= 1e6
    return v

# -------------------------------------------------------------------------
# Cell runner
# -------------------------------------------------------------------------
def run_cell(bench: str, model: str, size: str, args: list[str], writer):
    path = SRC / f"{bench}-{model}"
    if not path.is_dir():
        writer.writerow([bench, model, size, args[0] if args else "",
                         "", "", "-1", "missing"])
        return

    # Build (only if binary/target isn't already there for compiled targets)
    cmds = build_cmd(model, path)
    if cmds:
        # Check if we've already built
        need_build = False
        if model in ("cuda", "omp", "serial"):
            need_build = not (path / "main").exists()
        elif model == "rust":
            d = path / "target" / "release"
            need_build = not d.is_dir() or not any(
                f.is_file() and os.access(f, os.X_OK) and not f.suffix
                and f.stat().st_size > 100000 for f in d.iterdir())
        if need_build:
            for c in cmds:
                r = subprocess.run(c, cwd=path, env=ENV,
                                   capture_output=True, text=True,
                                   timeout=600)
                if r.returncode != 0:
                    writer.writerow([bench, model, size,
                                     args[0] if args else "",
                                     "", "", str(r.returncode),
                                     "build_fail"])
                    return

    cmd = run_binary_cmd(model, path, args)
    if not cmd:
        writer.writerow([bench, model, size, args[0] if args else "",
                         "", "", "-1", "no_binary"])
        return

    to = timeout_for(model, size)
    t0 = time.time()
    try:
        r = subprocess.run(cmd, cwd=path, env=ENV, capture_output=True,
                           text=True, timeout=to)
        wall = time.time() - t0
        us = parse_time_us(r.stdout)
        note = "" if r.returncode == 0 else "nonzero_exit"
        if r.returncode == 0 and us is None:
            note = "no_time_line"
        writer.writerow([bench, model, size, args[0] if args else "",
                         f"{wall:.2f}",
                         f"{us:.3f}" if us is not None else "",
                         str(r.returncode), note])
    except subprocess.TimeoutExpired:
        wall = time.time() - t0
        writer.writerow([bench, model, size, args[0] if args else "",
                         f"{wall:.2f}", "", "-2", f"timeout_{to}s"])


# -------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benches", nargs="*", default=list(BENCHMARKS.keys()))
    ap.add_argument("--models",  nargs="*", default=MODELS)
    ap.add_argument("--sizes",   nargs="*", default=["small","medium","large"])
    ap.add_argument("--out", default=str(OUT_CSV))
    a = ap.parse_args()

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["bench","model","size","first_arg","wall_s",
                    "exec_time_us","exit_code","notes"])
        f.flush()
        for bench in a.benches:
            for size_label, args in BENCHMARKS[bench]:
                if size_label not in a.sizes:
                    continue
                for model in a.models:
                    print(f"  [{bench} {model} {size_label}]", flush=True)
                    run_cell(bench, model, size_label, args, w)
                    f.flush()
    print(f"Wrote {a.out}")

if __name__ == "__main__":
    main()
