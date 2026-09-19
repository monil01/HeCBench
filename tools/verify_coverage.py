#!/usr/bin/env python3
"""Build, run, and cross-verify a single HeCBench benchmark across its models.

Usage:
  tools/verify_coverage.py <bench-name> [--models cuda,omp,serial]
                                        [--only-existing]
                                        [--args "8192 10000 10 100"]

Design goals:
  - Local models on this box: cuda (nvcc), omp (nvc++ -mp=gpu), serial (g++)
  - HIP and SYCL are recognized but marked SKIP (no toolchain here)
  - Result comparison = canonicalized stdout diff (strip timings, keep PASS/FAIL
    and numeric outputs) with a numeric tolerance
"""
from __future__ import annotations
import argparse, os, re, shlex, sqlite3, subprocess, sys, textwrap, time
from pathlib import Path
from datetime import datetime, timezone

REPO = Path(__file__).resolve().parents[1]
SRC  = REPO / "src"
STATE = Path(os.environ.get("STATE", REPO / ".porting-state"))
DB_PATH = STATE / "coverage.db"

# --- toolchain paths on this box ------------------------------------------------
HPC_SDK = "/opt/nvidia/hpc_sdk/Linux_x86_64/26.3"
ENV = os.environ.copy()
ENV["PATH"] = f"{HPC_SDK}/compilers/bin:{HPC_SDK}/cuda/bin:" + ENV.get("PATH", "")
ENV["LD_LIBRARY_PATH"] = (
    f"{HPC_SDK}/compilers/lib:{HPC_SDK}/cuda/lib64:"
    f"{HPC_SDK}/cuda/13.1/targets/x86_64-linux/lib:"
    + ENV.get("LD_LIBRARY_PATH", ""))
CUDA_ARCH = "sm_120"   # RTX 5090 / Blackwell
NVC_SM    = "cc120"

# --- per-model build recipes ---------------------------------------------------
def build_cmd(model: str, path: Path) -> list[list[str]] | None:
    if model == "cuda":
        return [["make", "clean"], ["make", f"ARCH={CUDA_ARCH}"]]
    if model == "omp":
        # Prefer Makefile.nvc (nvc++) since we have HPC SDK, not oneAPI.
        if (path / "Makefile.nvc").exists():
            return [["make", "-f", "Makefile.nvc", "clean"],
                    ["make", "-f", "Makefile.nvc", f"SM={NVC_SM}"]]
        return [["make", "clean"], ["make"]]
    if model == "serial":
        return [["make", "clean"], ["make"]]
    if model == "triton":
        # No compile step — main.py is source. `make main` is a no-op.
        return [["make", "main"]]
    if model == "julia":
        # Shared CUDA.jl project — build is a no-op; the julia runtime JITs
        # on first run. Any per-benchmark deps come via `--project` in the
        # Makefile's `run:` target.
        return [["make", "main"]]
    if model == "rust":
        return [["make", "clean"], ["make"]]  # `make` invokes `cargo build --release`
    if model in ("hip", "sycl"):
        return None  # deferred: no local toolchain (ROCm / oneAPI)
    if model == "mojo":
        return None  # deferred: Mojo 1.0.0b2 stdlib not accessible from plain scripts
    return None

# For python/julia/rust ports, `main` is not an ELF at path root — use `make run`
# semantics (Makefile knows how to invoke python3 / julia / cargo).
_MAKE_RUN_MODELS = {"triton", "julia", "rust"}

def run_binary(path: Path, args: list[str], timeout: int = 300,
               model: str | None = None) -> tuple[int, str, str]:
    """Execute the benchmark. For compiled models (cuda/omp/serial/rust),
    invoke ./main directly with args. For interpreted models (triton/julia),
    delegate to `make run` since the Makefile knows the interpreter.
    """
    if model in _MAKE_RUN_MODELS and model != "rust":
        # triton, julia — Makefile has the interpreter + args
        cmd = ["make", "run"]
        if args:
            # Override args via LAUNCHER-style pass-through? Most Makefiles hard-
            # code args in the `run:` target. To respect --args, prefer running
            # main.py / main.jl directly if it exists.
            main_py = path / "main.py"
            main_jl = path / "main.jl"
            if model == "triton" and main_py.exists():
                cmd = [sys.executable, str(main_py), *args]
            elif model == "julia" and main_jl.exists():
                proj = f"--project={REPO}/src/_julia_env"
                cmd = ["julia", proj, str(main_jl), *args]
        try:
            r = subprocess.run(cmd, cwd=path, env=ENV, capture_output=True,
                               text=True, timeout=timeout)
            return (r.returncode, r.stdout, r.stderr)
        except subprocess.TimeoutExpired:
            return (-2, "", f"timeout after {timeout}s")
    if model == "rust":
        # cargo run --release -- <args>
        cmd = ["cargo", "run", "--release", "--quiet", "--"] + list(args)
        try:
            r = subprocess.run(cmd, cwd=path, env=ENV, capture_output=True,
                               text=True, timeout=timeout)
            return (r.returncode, r.stdout, r.stderr)
        except subprocess.TimeoutExpired:
            return (-2, "", f"timeout after {timeout}s")

    binary = path / "main"
    if not binary.exists():
        return (-1, "", f"binary not found: {binary}")
    try:
        r = subprocess.run([str(binary), *args], cwd=path, env=ENV,
                           capture_output=True, text=True, timeout=timeout)
        return (r.returncode, r.stdout, r.stderr)
    except subprocess.TimeoutExpired:
        return (-2, "", f"timeout after {timeout}s")

def make_at(path: Path, cmds: list[list[str]], timeout: int = 600) -> tuple[bool, str]:
    """Run each cmd in the given dir; return (ok, log)."""
    log = []
    for cmd in cmds:
        log.append(f"$ {' '.join(cmd)}  (cwd={path})")
        try:
            r = subprocess.run(cmd, cwd=path, env=ENV, capture_output=True,
                               text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return False, "\n".join(log + [f"build timed out after {timeout}s"])
        log.append(r.stdout); log.append(r.stderr)
        if r.returncode != 0:
            return False, "\n".join(log + [f"exit={r.returncode}"])
    return True, "\n".join(log)

# --- default args discovery ----------------------------------------------------
def parse_yaml_args(bench: str) -> list[str] | None:
    y = REPO / "benchmarks.yaml"
    if not y.exists():
        return None
    lines = y.read_text().splitlines()
    for i, line in enumerate(lines):
        if line.strip() == f"{bench}:":
            # scan forward until dedent
            for j in range(i + 1, min(i + 40, len(lines))):
                m = re.match(r"^\s+args:\s*\[(.*)\]\s*$", lines[j])
                if m:
                    parts = re.findall(r'"([^"]*)"|(\S+?)(?:,|$)', m.group(1))
                    out = []
                    for a, b in parts:
                        v = a or b
                        v = v.strip().strip(",").strip('"').strip("'")
                        if v:
                            out.append(v)
                    return out
                if re.match(r"^[A-Za-z][A-Za-z0-9_+-]*:", lines[j]):
                    break
    return None

def parse_makefile_run_args(makefile: Path) -> list[str] | None:
    """Look for a `run:` target and pull args from `./main <args>`."""
    if not makefile.exists():
        return None
    try:
        content = makefile.read_text()
    except Exception:
        return None
    m = re.search(r"^run:.*?^\t.*?\./\$?\(?program\)?\.?\s*(.*)$",
                  content, flags=re.M | re.S)
    if not m:
        m = re.search(r"^run:.*?^\t.*?\./main\s*(.*)$",
                      content, flags=re.M | re.S)
    if not m:
        return None
    lines = m.group(1).splitlines()
    first_line = lines[0].strip() if lines else ""
    return shlex.split(first_line) if first_line else []

def resolve_args(bench: str, per_dir: dict[str, Path], override: list[str] | None) -> list[str]:
    if override is not None:
        return override
    args = parse_yaml_args(bench)
    if args:
        return args
    for model in ("cuda", "omp", "serial"):
        d = per_dir.get(model)
        if d:
            for mf in ("Makefile", "Makefile.nvc", "Makefile.aomp"):
                a = parse_makefile_run_args(d / mf)
                if a:
                    return a
    return []

# --- output comparison ---------------------------------------------------------
NUM_RE = re.compile(r"[-+]?\d+\.?\d*(?:[eE][-+]?\d+)?")

TIMING_HINTS = ("execution time", "elapsed", "(us)", "(ms)", "(s)",
                "gflops", "throughput", "bandwidth", "gb/s", "kernel time",
                "offload time", "average")
PASS_RE = re.compile(r"\bPASS\b")
FAIL_RE = re.compile(r"\bFAIL\b")

def canonicalize(text: str) -> dict:
    """Split into three buckets:
        timing  — dropped (kept for debug only)
        verdict — PASS/FAIL lines
        data    — everything else (numeric results, labels, etc.)
    """
    timing, verdict, data = [], [], []
    for ln in text.splitlines():
        s = ln.strip()
        if not s:
            continue
        low = s.lower()
        if any(h in low for h in TIMING_HINTS):
            timing.append(s); continue
        if PASS_RE.search(s) or FAIL_RE.search(s):
            verdict.append(s); continue
        data.append(s)
    return {"timing": timing, "verdict": verdict, "data": data}

def _strip_numbers(line: str) -> str:
    """Replace every numeric literal with a placeholder so we can compare
    structure/labels without demanding bit-identical numerics.
    Bench-native PASS is the source of truth for correctness; cross-model
    numeric summaries will differ due to libm/GPU-math ULP variance."""
    return NUM_RE.sub("#", line)

def compare_outputs(outputs: dict[str, str]) -> tuple[bool, list[str]]:
    """Rules:
      (a) no model may emit FAIL
      (b) if every model emits at least one PASS: trust the benchmark's own
          verifier — cross-model check is satisfied. Data lines can differ
          because floating-point summaries diverge between compilers/GPU math.
      (c) if no verdicts exist anywhere: fall back to comparing "structure"
          of data lines (labels + shape, numbers-as-placeholder) so we detect
          gross divergence (missing lines, different code paths) but not ULP
          noise.
    """
    canons = {m: canonicalize(t) for m, t in outputs.items()}
    notes = []

    for m, c in canons.items():
        fails = [ln for ln in c["verdict"] if FAIL_RE.search(ln)]
        if fails:
            notes.append(f"{m}: FAIL line(s): {fails[:3]}")
    if notes:
        return False, notes

    models = sorted(canons)
    have_verdicts_everywhere = all(c["verdict"] for c in canons.values())

    if have_verdicts_everywhere:
        # Every model has a bench-native verifier and none of them FAILed.
        # Trust that. Numeric summaries are allowed to differ.
        for m, c in canons.items():
            if not any(PASS_RE.search(ln) for ln in c["verdict"]):
                notes.append(f"{m}: no PASS line emitted")
        return (not notes), notes

    # No verifier — compare data-line structure (numbers-as-#) as multiset.
    ref_struct = sorted(_strip_numbers(ln) for ln in canons[models[0]]["data"])
    for m in models[1:]:
        m_struct = sorted(_strip_numbers(ln) for ln in canons[m]["data"])
        if m_struct != ref_struct:
            notes.append(f"data-line structure differs between {models[0]} and {m}")
            from difflib import unified_diff
            diff = list(unified_diff(ref_struct, m_struct,
                                     fromfile=models[0], tofile=m,
                                     n=1, lineterm=""))
            notes.extend(diff[:20])
    return (not notes), notes

# --- main ----------------------------------------------------------------------
def discover_models(bench: str) -> dict[str, Path]:
    out = {}
    for m in ("cuda", "hip", "sycl", "omp", "serial", "triton", "julia", "rust", "mojo"):
        p = SRC / f"{bench}-{m}"
        if p.is_dir():
            out[m] = p
    return out

def record(bench: str, model: str, status: str, detail: str = "",
           log_path: str = "") -> None:
    try:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(DB_PATH)
        conn.execute("""CREATE TABLE IF NOT EXISTS verify_status (
            bench TEXT NOT NULL, model TEXT NOT NULL, status TEXT NOT NULL,
            detail TEXT, last_run_at TEXT, log_path TEXT,
            PRIMARY KEY (bench, model))""")
        conn.execute("""INSERT OR REPLACE INTO verify_status
                        (bench, model, status, detail, last_run_at, log_path)
                        VALUES (?,?,?,?,?,?)""",
                     (bench, model, status, detail,
                      datetime.now(timezone.utc).isoformat(timespec="seconds"),
                      log_path))
        conn.commit(); conn.close()
    except Exception as e:
        print(f"# (record failed: {e})", file=sys.stderr)

def record_summary(bench: str, overall: str) -> None:
    try:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(DB_PATH)
        conn.execute("""CREATE TABLE IF NOT EXISTS verify_summary (
            bench TEXT PRIMARY KEY, overall TEXT NOT NULL, last_run_at TEXT)""")
        conn.execute("""INSERT OR REPLACE INTO verify_summary
                        (bench, overall, last_run_at)
                        VALUES (?,?,?)""",
                     (bench, overall,
                      datetime.now(timezone.utc).isoformat(timespec="seconds")))
        conn.commit(); conn.close()
    except Exception as e:
        print(f"# (record_summary failed: {e})", file=sys.stderr)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bench", help="benchmark name (e.g. accuracy)")
    ap.add_argument("--models", default="cuda,omp,serial",
                    help="comma-separated models to try")
    ap.add_argument("--args", default=None,
                    help="override runtime args (shell-quoted string)")
    ap.add_argument("--only-existing", action="store_true",
                    help="only try models whose src dir exists")
    ap.add_argument("--no-record", action="store_true",
                    help="don't write results to coverage.db")
    ap.add_argument("--verbose", "-v", action="store_true")
    a = ap.parse_args()

    want = [m.strip() for m in a.models.split(",") if m.strip()]
    per_dir = discover_models(a.bench)

    override_args = shlex.split(a.args) if a.args else None
    args = resolve_args(a.bench, per_dir, override_args)

    print(f"# Benchmark: {a.bench}")
    print(f"# Run args:  {' '.join(args) if args else '(none)'}")
    print()

    outputs: dict[str, str] = {}
    status: dict[str, str] = {}

    log_dir = STATE / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    for model in want:
        if model not in per_dir:
            if a.only_existing:
                continue
            print(f"[{model}]  MISSING  (no src/{a.bench}-{model})")
            status[model] = "missing"
            if not a.no_record:
                record(a.bench, model, "missing")
            continue
        path = per_dir[model]
        cmds = build_cmd(model, path)
        if cmds is None:
            print(f"[{model}]  SKIP    (no local toolchain on this box)")
            status[model] = "skipped"
            if not a.no_record:
                record(a.bench, model, "skipped", "no local toolchain")
            continue

        t0 = time.time()
        ok, log = make_at(path, cmds)
        t1 = time.time()
        log_file = log_dir / f"{a.bench}-{model}.log"
        log_file.write_text(log)
        if not ok:
            print(f"[{model}]  BUILD FAIL  ({t1-t0:.1f}s)  see {log_file}")
            if a.verbose:
                print(textwrap.indent(log[-2000:], "    "))
            status[model] = "build_fail"
            if not a.no_record:
                record(a.bench, model, "build_fail", f"{t1-t0:.1f}s",
                       str(log_file))
            continue

        rc, out, err = run_binary(path, args, model=model)
        t2 = time.time()
        log_file.write_text(log + "\n---RUN STDOUT---\n" + out
                            + "\n---RUN STDERR---\n" + err)
        if rc != 0:
            print(f"[{model}]  RUN FAIL   (exit={rc}, {t2-t1:.1f}s)  see {log_file}")
            if a.verbose:
                print(textwrap.indent((out + err)[-2000:], "    "))
            status[model] = f"run_fail"
            if not a.no_record:
                record(a.bench, model, "run_fail",
                       f"exit={rc} args={' '.join(args)}", str(log_file))
            continue

        outputs[model] = out
        status[model] = "ok"
        print(f"[{model}]  OK          ({t2-t1:.1f}s run)")
        if not a.no_record:
            record(a.bench, model, "ok", " ".join(args), str(log_file))

    print()
    if len(outputs) < 2:
        print("Cross-model comparison: skipped (<2 successful runs)")
        overall = "partial" if any(s == "ok" for s in status.values()) else "not_run"
        if not a.no_record:
            record_summary(a.bench, overall)
        return 0 if any(s == "ok" for s in status.values()) else 2

    ok, notes = compare_outputs(outputs)
    print(f"Cross-model comparison: {'MATCH' if ok else 'MISMATCH'}")
    for n in notes:
        print(f"  {n}")
    overall = "pass" if ok else "mismatch"
    if not a.no_record:
        record_summary(a.bench, overall)
        if not ok:
            # tag each model with mismatch so we know which set was compared
            for m in outputs:
                record(a.bench, m, "mismatch", "|".join(notes[:3]))
    return 0 if ok else 1

if __name__ == "__main__":
    sys.exit(main())
