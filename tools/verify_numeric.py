#!/usr/bin/env python3
"""Numerical accuracy checker for HeCBench Julia ports.

This tool is intentionally stricter than tools/verify_coverage.py for
benchmarks that do not emit PASS/FAIL. It records whether a result was
accepted because the benchmark's native verifier passed, because comparable
numeric stdout matched within tolerance, or because numeric checking could
not be performed.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
STATE = Path(os.environ.get("STATE", REPO / ".porting-state"))
DB_PATH = STATE / "numeric_accuracy.db"
NUMERIC_DIR = STATE / "numeric_accuracy"

sys.path.insert(0, str(REPO / "tools"))
import verify_coverage as cov  # noqa: E402


NUM_RE = cov.NUM_RE
PASS_RE = re.compile(r"\bPASSED?\b", re.IGNORECASE)
FAIL_RE = re.compile(r"\bFAILED?\b", re.IGNORECASE)
TIMING_HINTS = cov.TIMING_HINTS


def parse_list_value(value: str) -> list[str]:
    parts = re.findall(r'"([^"]*)"|(\S+?)(?:,|$)', value)
    out = []
    for quoted, bare in parts:
        item = quoted or bare
        item = item.strip().strip(",").strip('"').strip("'")
        if item:
            out.append(item)
    return out


def parse_numeric_check(bench: str) -> dict:
    y = REPO / "benchmarks.yaml"
    if not y.exists():
        return {}
    lines = y.read_text().splitlines()
    out: dict[str, object] = {}
    for i, line in enumerate(lines):
        if line.strip() != f"{bench}:":
            continue
        in_numeric = False
        in_stdout_regex = False
        stdout_regex: list[str] = []
        for raw in lines[i + 1 : min(i + 80, len(lines))]:
            if re.match(r"^[A-Za-z][A-Za-z0-9_+-]*:", raw):
                break
            if re.match(r"^\s+numeric_check:\s*$", raw):
                in_numeric = True
                continue
            if not in_numeric:
                continue
            if in_stdout_regex:
                m = re.match(r"^\s+-\s+['\"]?(.*?)['\"]?\s*$", raw)
                if m:
                    stdout_regex.append(m.group(1))
                    continue
                in_stdout_regex = False
            m = re.match(r"^\s+stdout_regex:\s*\[(.*)\]\s*$", raw)
            if m:
                stdout_regex.extend(parse_list_value(m.group(1)))
                continue
            if re.match(r"^\s+stdout_regex:\s*$", raw):
                in_stdout_regex = True
                continue
            m = re.match(r"^\s+args:\s*\[(.*)\]\s*$", raw)
            if m:
                out["args"] = parse_list_value(m.group(1))
                continue
            m = re.match(r"^\s+timeout:\s*([0-9]+)\s*$", raw)
            if m:
                out["timeout"] = int(m.group(1))
                continue
            m = re.match(r"^\s+rtol:\s*([0-9eE.+-]+)\s*$", raw)
            if m:
                out["rtol"] = float(m.group(1))
                continue
            m = re.match(r"^\s+atol:\s*([0-9eE.+-]+)\s*$", raw)
            if m:
                out["atol"] = float(m.group(1))
                continue
        if stdout_regex:
            out["stdout_regex"] = stdout_regex
        break
    return out


def numeric_db() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS numeric_status (
            bench TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            method TEXT NOT NULL,
            detail TEXT,
            max_abs_error REAL,
            max_rel_error REAL,
            checked_values INTEGER,
            last_run_at TEXT NOT NULL,
            log_path TEXT
        )"""
    )
    return conn


def record(
    bench: str,
    status: str,
    method: str,
    detail: str = "",
    max_abs_error: float | None = None,
    max_rel_error: float | None = None,
    checked_values: int = 0,
    log_path: str = "",
) -> None:
    conn = numeric_db()
    conn.execute(
        """INSERT OR REPLACE INTO numeric_status
           (bench, status, method, detail, max_abs_error, max_rel_error,
            checked_values, last_run_at, log_path)
           VALUES (?,?,?,?,?,?,?,?,?)""",
        (
            bench,
            status,
            method,
            detail,
            max_abs_error,
            max_rel_error,
            checked_values,
            datetime.now(timezone.utc).isoformat(timespec="seconds"),
            log_path,
        ),
    )
    conn.commit()
    conn.close()


def canonicalize(text: str) -> dict:
    timing, verdict, data = [], [], []
    for ln in text.splitlines():
        s = ln.strip()
        if not s:
            continue
        low = s.lower()
        if any(h in low for h in TIMING_HINTS):
            timing.append(s)
            continue
        if PASS_RE.search(s) or FAIL_RE.search(s):
            verdict.append(s)
            continue
        data.append(s)
    return {"timing": timing, "verdict": verdict, "data": data}


def non_timing_data_lines(text: str) -> list[str]:
    can = canonicalize(text)
    out = []
    for line in can["data"]:
        low = line.lower()
        if re.search(r"\b(?:us|ms|sec|secs|seconds)\b", low):
            continue
        if not NUM_RE.search(line):
            continue
        stripped = line.strip()
        if ":" not in stripped and "=" not in stripped and not re.match(r"^[-+]?\d", stripped):
            continue
        out.append(line)
    return out


def line_shape(line: str) -> str:
    return NUM_RE.sub("#", line)


def numbers(line: str) -> list[float]:
    out = []
    for token in NUM_RE.findall(line):
        try:
            out.append(float(token))
        except ValueError:
            pass
    return out


def configured_stdout_values(text: str, patterns: list[str]) -> tuple[list[float], list[str]]:
    values: list[float] = []
    missing: list[str] = []
    for pattern in patterns:
        rx = re.compile(pattern)
        matched = False
        for line in text.splitlines():
            m = rx.search(line)
            if not m:
                continue
            matched = True
            groups = m.groups() or (m.group(0),)
            for group in groups:
                try:
                    values.append(float(group))
                except ValueError:
                    values.extend(numbers(group))
        if not matched:
            missing.append(pattern)
    return values, missing


def compare_values(
    ref_nums: list[float],
    got_nums: list[float],
    rtol: float,
    atol: float,
) -> tuple[bool, str, float, float, int]:
    if len(ref_nums) != len(got_nums):
        return False, f"configured stdout numeric count differs: cuda={len(ref_nums)} julia={len(got_nums)}", math.nan, math.nan, 0

    max_abs = 0.0
    max_rel = 0.0
    for i, (a, b) in enumerate(zip(ref_nums, got_nums), start=1):
        abs_err = abs(a - b)
        rel_err = abs_err / max(abs(a), atol)
        max_abs = max(max_abs, abs_err)
        max_rel = max(max_rel, rel_err)
        if abs_err > atol + rtol * abs(a):
            return (
                False,
                f"configured stdout value {i} differs: cuda={a} julia={b} abs={abs_err} rel={rel_err}",
                max_abs,
                max_rel,
                i,
            )
    if not ref_nums:
        return False, "configured stdout regex matched no numeric values", math.nan, math.nan, 0
    return True, "configured stdout matched", max_abs, max_rel, len(ref_nums)


def compare_numeric_stdout(
    ref_text: str,
    got_text: str,
    rtol: float,
    atol: float,
) -> tuple[bool, str, float, float, int]:
    ref_lines = non_timing_data_lines(ref_text)
    got_lines = non_timing_data_lines(got_text)
    if len(ref_lines) != len(got_lines):
        return False, f"data line count differs: cuda={len(ref_lines)} julia={len(got_lines)}", math.nan, math.nan, 0

    max_abs = 0.0
    max_rel = 0.0
    checked = 0
    for i, (ref, got) in enumerate(zip(ref_lines, got_lines), start=1):
        if line_shape(ref) != line_shape(got):
            return False, f"line {i} shape differs: cuda={ref!r} julia={got!r}", max_abs, max_rel, checked
        ref_nums = numbers(ref)
        got_nums = numbers(got)
        if len(ref_nums) != len(got_nums):
            return False, f"line {i} numeric count differs", max_abs, max_rel, checked
        for j, (a, b) in enumerate(zip(ref_nums, got_nums), start=1):
            abs_err = abs(a - b)
            rel_err = abs_err / max(abs(a), atol)
            max_abs = max(max_abs, abs_err)
            max_rel = max(max_rel, rel_err)
            checked += 1
            if abs_err > atol + rtol * abs(a):
                return (
                    False,
                    f"line {i} value {j} differs: cuda={a} julia={b} abs={abs_err} rel={rel_err}",
                    max_abs,
                    max_rel,
                    checked,
                )

    if checked == 0:
        return False, "no comparable numeric stdout values", math.nan, math.nan, 0
    return True, "numeric stdout matched", max_abs, max_rel, checked


def run_model(bench: str, model: str, args: list[str], build: bool, timeout: int) -> tuple[str, str, str, str]:
    per_dir = cov.discover_models(bench)
    if model not in per_dir:
        return "missing", "", "", f"missing src/{bench}-{model}"
    path = per_dir[model]
    log_chunks = []
    if build:
        cmds = cov.build_cmd(model, path)
        if cmds is None:
            return "skipped", "", "", "no local toolchain"
        ok, build_log = cov.make_at(path, cmds)
        log_chunks.append(build_log)
        if not ok:
            return "build_fail", "", "", build_log
    rc, out, err = cov.run_binary(path, args, timeout=timeout, model=model)
    log_chunks.append("---RUN STDOUT---\n" + out + "\n---RUN STDERR---\n" + err)
    if rc != 0:
        return "run_fail", out, err, "\n".join(log_chunks + [f"exit={rc}"])
    return "ok", out, err, "\n".join(log_chunks)


def verify_one(bench: str, args_override: list[str] | None, rtol: float, atol: float, timeout: int, build: bool) -> int:
    per_dir = cov.discover_models(bench)
    numeric_cfg = parse_numeric_check(bench)
    if args_override is not None:
        args = args_override
    elif "args" in numeric_cfg:
        args = list(numeric_cfg["args"])
    else:
        args = cov.resolve_args(bench, per_dir, None)
    timeout = int(numeric_cfg.get("timeout", timeout))
    rtol = float(numeric_cfg.get("rtol", rtol))
    atol = float(numeric_cfg.get("atol", atol))
    run_dir = NUMERIC_DIR / "logs"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / f"{bench}-numeric.log"

    status_cuda, out_cuda, err_cuda, log_cuda = run_model(bench, "cuda", args, build, timeout)
    status_julia, out_julia, err_julia, log_julia = run_model(bench, "julia", args, build, timeout)
    log_path.write_text(
        f"# Benchmark: {bench}\n# Args: {' '.join(args) if args else '(none)'}\n\n"
        f"## cuda status={status_cuda}\n{log_cuda}\n\n"
        f"## julia status={status_julia}\n{log_julia}\n"
    )

    if status_cuda != "ok" or status_julia != "ok":
        detail = f"cuda={status_cuda} julia={status_julia}"
        record(bench, "not_run", "run_status", detail, log_path=str(log_path))
        print(f"{bench}: NOT_RUN {detail}")
        return 2

    can_cuda = canonicalize(out_cuda)
    can_julia = canonicalize(out_julia)
    if any(FAIL_RE.search(line) for line in can_cuda["verdict"]):
        detail = "CUDA reference emitted FAIL verdict"
        record(bench, "not_run", "reference_fail", detail, log_path=str(log_path))
        print(f"{bench}: NOT_RUN {detail}")
        return 2

    if any(FAIL_RE.search(line) for line in can_julia["verdict"]):
        detail = "Julia emitted FAIL verdict"
        record(bench, "mismatch", "native_verdict", detail, log_path=str(log_path))
        print(f"{bench}: MISMATCH {detail}")
        return 1

    if can_cuda["verdict"] and can_julia["verdict"]:
        if any(PASS_RE.search(line) for line in can_cuda["verdict"]) and any(PASS_RE.search(line) for line in can_julia["verdict"]):
            record(bench, "pass", "native_pass_trusted", "both CUDA and Julia emitted PASS", checked_values=0, log_path=str(log_path))
            print(f"{bench}: PASS native_pass_trusted")
            return 0
        detail = f"verdicts without PASS: cuda={can_cuda['verdict']} julia={can_julia['verdict']}"
        record(bench, "mismatch", "native_verdict", detail, log_path=str(log_path))
        print(f"{bench}: MISMATCH {detail}")
        return 1

    if "stdout_regex" in numeric_cfg:
        patterns = list(numeric_cfg["stdout_regex"])
        ref_nums, ref_missing = configured_stdout_values(out_cuda, patterns)
        got_nums, got_missing = configured_stdout_values(out_julia, patterns)
        if ref_missing or got_missing:
            detail = f"configured stdout regex missing: cuda={ref_missing} julia={got_missing}"
            record(bench, "mismatch", "numeric_stdout", detail, checked_values=0, log_path=str(log_path))
            print(f"{bench}: MISMATCH {detail}")
            return 1
        ok, detail, max_abs, max_rel, checked = compare_values(ref_nums, got_nums, rtol, atol)
    else:
        ok, detail, max_abs, max_rel, checked = compare_numeric_stdout(out_cuda, out_julia, rtol, atol)
    if ok:
        record(bench, "pass", "numeric_stdout", detail, max_abs, max_rel, checked, str(log_path))
        print(f"{bench}: PASS numeric_stdout values={checked} max_abs={max_abs:.3g} max_rel={max_rel:.3g}")
        return 0

    if checked == 0 and detail.startswith("no comparable numeric"):
        record(bench, "unverified", "no_numeric_data", detail, max_abs, max_rel, checked, str(log_path))
        print(f"{bench}: UNVERIFIED {detail}")
        return 1

    record(bench, "mismatch", "numeric_stdout", detail, max_abs, max_rel, checked, str(log_path))
    print(f"{bench}: MISMATCH {detail}")
    return 1


def summarize() -> None:
    conn = numeric_db()
    rows = conn.execute(
        "SELECT status, method, COUNT(*) FROM numeric_status GROUP BY status, method ORDER BY status, method"
    ).fetchall()
    reports_dir = NUMERIC_DIR / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "database": str(DB_PATH),
        "counts": [{"status": s, "method": m, "count": c} for s, m, c in rows],
    }
    out = NUMERIC_DIR / "numeric_accuracy_summary.json"
    out.write_text(json.dumps(summary, indent=2) + "\n")
    (reports_dir / "counts.txt").write_text(
        "".join(f"{s}|{m}|{c}\n" for s, m, c in rows)
    )
    report_queries = {
        "mismatches.txt": "SELECT bench FROM numeric_status WHERE status='mismatch' ORDER BY bench",
        "unverified.txt": "SELECT bench FROM numeric_status WHERE status='unverified' ORDER BY bench",
        "not_run.txt": "SELECT bench FROM numeric_status WHERE status='not_run' ORDER BY bench",
    }
    for filename, query in report_queries.items():
        names = [row[0] for row in conn.execute(query)]
        (reports_dir / filename).write_text("".join(f"{name}\n" for name in names))
    for s, m, c in rows:
        print(f"{s}|{m}|{c}")
    conn.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("bench", nargs="?", help="benchmark name")
    ap.add_argument("--bench-list", help="file containing one benchmark per line")
    ap.add_argument("--args", default=None, help="override runtime args")
    ap.add_argument("--rtol", type=float, default=1e-5)
    ap.add_argument("--atol", type=float, default=1e-8)
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--no-build", action="store_true")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--summary", action="store_true")
    ns = ap.parse_args()

    if ns.summary:
        summarize()
        return 0

    benches: list[str] = []
    if ns.bench:
        benches.append(ns.bench)
    if ns.bench_list:
        benches.extend(
            line.strip()
            for line in Path(ns.bench_list).read_text().splitlines()
            if line.strip() and not line.startswith("#")
        )
    if not benches:
        ap.error("provide a bench or --bench-list")

    args_override = cov.shlex.split(ns.args) if ns.args else None
    rc = 0
    if ns.skip_existing:
        conn = numeric_db()
        done = {row[0] for row in conn.execute("SELECT bench FROM numeric_status")}
        conn.close()
        benches = [b for b in benches if b not in done]

    for bench in benches:
        rc = max(rc, verify_one(bench, args_override, ns.rtol, ns.atol, ns.timeout, not ns.no_build))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
