#!/usr/bin/env python3
"""Report CUDA-to-IRIS directory coverage and local validation evidence."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import subprocess


def inventory(repo, state):
    cuda = {p.name[:-5] for p in (repo / "src").glob("*-cuda") if p.is_dir()}
    iris = {p.name[:-5] for p in (repo / "src").glob("*-iris") if p.is_dir()}
    tracked = subprocess.run(
        ["git", "ls-tree", "-r", "--name-only", "HEAD", "--", "src"],
        cwd=repo, capture_output=True, text=True, check=True,
    ).stdout.splitlines()
    committed_files = set(tracked)
    rows = []
    for bench in sorted(cuda & iris):
        port = repo / "src" / (bench + "-iris")
        manifest = state / "porting_logs" / (bench + "-iris") / "manifest.json"
        status = "no_manifest"
        evidence = {}
        if manifest.exists():
            try:
                evidence = json.loads(manifest.read_text())
                status = evidence.get("status", "unknown")
            except (ValueError, OSError):
                status = "invalid_manifest"
        required = ["main.cpp", "kernel.hip.cpp", "Makefile"]
        missing = [name for name in required if not (port / name).is_file()]
        committed = all(f"src/{bench}-iris/{name}" in committed_files
                        for name in required)
        rows.append({"bench": bench, "status": status,
                     "committed_layout": committed, "missing_files": missing,
                     "runtime_env": evidence.get("runtime_env"),
                     "accuracy_method": evidence.get("accuracy_method")})
    return {"cuda_total": len(cuda), "iris_directories": len(iris),
            "covered_cuda": len(cuda & iris),
            "directory_coverage_percent": round(100 * len(cuda & iris) / len(cuda), 2)
                if cuda else 0,
            "missing_cuda": sorted(cuda - iris),
            "status_counts": dict(Counter(row["status"] for row in rows)),
            "committed_complete_layouts": sum(row["committed_layout"] for row in rows),
            "ports": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--baseline", type=int)
    parser.add_argument("--additional", type=int, default=100)
    parser.add_argument("--batch-plan", type=Path,
                        help="include worker progress from a coordinator plan JSON")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    state = Path(os.environ.get("PORTING_STATE_DIR",
                               os.environ.get("STATE", str(repo / ".porting-state"))))
    result = inventory(repo, state)
    if args.batch_plan:
        plan = json.loads(args.batch_plan.read_text())
        handled = {}
        workers = []
        for worker in plan["workers"]:
            checkpoint = state / "porting_logs" / (worker["bench"] + "-iris") / "progress.json"
            progress = json.loads(checkpoint.read_text()) if checkpoint.exists() else {}
            workers.append({"worker": worker["bench"],
                            "current": progress.get("current"),
                            "remaining": len(progress.get("remaining", []))})
            for row in progress.get("completed", []):
                handled[row["bench"]] = row
        inventory_rows = {row["bench"]: row for row in result["ports"]}
        verified = [bench for bench, row in handled.items()
                    if row.get("commit") and bench in inventory_rows
                    and inventory_rows[bench]["committed_layout"]
                    and inventory_rows[bench]["status"] == "ok"]
        result["batch"] = {"handled": len(handled), "verified_committed": len(verified),
                           "verified_benchmarks": sorted(verified), "workers": workers}
    if args.baseline is not None:
        result["batch_directory_additions"] = result["covered_cuda"] - args.baseline
        result["batch_directory_target"] = args.baseline + args.additional
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"CUDA: {result['cuda_total']}; IRIS siblings: {result['covered_cuda']} "
              f"({result['directory_coverage_percent']}%); "
              f"missing: {len(result['missing_cuda'])}")
        print(f"Committed complete layouts: {result['committed_complete_layouts']}")
        print("Manifest statuses: " + json.dumps(result["status_counts"], sort_keys=True))
        if args.baseline is not None:
            print(f"Directory additions: {result['batch_directory_additions']}/"
                  f"{args.additional}; target: {result['batch_directory_target']}")
        if "batch" in result:
            batch = result["batch"]
            print(f"Worker batch: {batch['verified_committed']} verified committed ports; "
                  f"{batch['handled']} handled assignments")
            for worker in batch["workers"]:
                print(f"  {worker['worker']}: current={worker['current']}, "
                      f"remaining={worker['remaining']}")
        print("Directory coverage does not imply successful HIP validation.")


if __name__ == "__main__":
    main()
