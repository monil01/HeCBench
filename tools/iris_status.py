#!/usr/bin/env python3
"""Report CUDA-to-IRIS directory coverage and local validation evidence."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import posixpath
import re
import subprocess


def vendor_bridge_layout(read_source):
    """Recognize the built shared bridge, not a file merely named vendor.cpp.

    This is a source-layout check, not proof of numerical validation. The reader
    is supplied separately for working files and HEAD blobs so uncommitted
    scaffolding cannot make a committed layout complete.
    """
    main = read_source("main.cpp") or ""
    main = re.sub(r"/\*.*?\*/|//[^\n]*", "", main, flags=re.S)
    library = re.search(r'"IRIS_KERNEL_HOST2HIP"\s*,\s*"([^"\n]+)"', main)
    if not (library and "hecbench_vendor_callback" in main and
            re.search(r'\bkernel\s*\(\s*"hecbench_vendor"\s*,', main)):
        return False
    shared = "../../tools/iris_vendor_bridge.cpp"
    if read_source(shared) is None or read_source("../../tools/iris_vendor_bridge.hpp") is None:
        return False
    makefile = (read_source("Makefile") or "").replace("\\\n", " ")
    # A real shared-library rule must compile the implementation (directly or
    # through an include wrapper), not merely list it as an unused dependency.
    rules = re.finditer(r"^([^\t#\n][^:\n]*):([^\n]*)\n((?:\t[^\n]*\n?)+)",
                        makefile, flags=re.M)
    for rule in rules:
        targets, dependencies, recipe = rule.groups()
        recipe = "\n".join(line.split("#", 1)[0] for line in recipe.splitlines())
        if library.group(1).removeprefix("./") not in targets.split() or "-shared" not in recipe:
            continue
        dependencies = dependencies.split("#", 1)[0].split()
        for index, source in enumerate(dependencies):
            if not source.endswith((".cpp", ".cc", ".cxx")):
                continue
            text = read_source(source)
            if text is None:
                continue
            text = re.sub(r"/\*.*?\*/|//[^\n]*", "", text, flags=re.S)
            includes_bridge = re.search(
                r'^\s*#\s*include\s*"\.\./\.\./tools/iris_vendor_bridge\.cpp"', text, flags=re.M)
            compiled = source in recipe or "$^" in recipe or (index == 0 and "$<" in recipe)
            if compiled and (source == shared or includes_bridge):
                return True
    return False


def inventory(repo, state):
    cuda = {p.name[:-5] for p in (repo / "src").glob("*-cuda") if p.is_dir()}
    iris = {p.name[:-5] for p in (repo / "src").glob("*-iris") if p.is_dir()}
    tracked = subprocess.run(
        ["git", "ls-tree", "-r", "--name-only", "HEAD", "--", "src",
         "tools/iris_vendor_bridge.cpp", "tools/iris_vendor_bridge.hpp"],
        cwd=repo, capture_output=True, text=True, check=True,
    ).stdout.splitlines()
    committed_files = set(tracked)
    committed_sources = {}

    def read_committed(path):
        if path not in committed_files:
            return None
        if path not in committed_sources:
            committed_sources[path] = subprocess.run(
                ["git", "show", f"HEAD:{path}"], cwd=repo,
                capture_output=True, text=True, check=True,
            ).stdout
        return committed_sources[path]

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
        required = ["main.cpp", "Makefile"]
        missing = [name for name in required if not (port / name).is_file()]
        prefix = f"src/{bench}-iris/"

        def working_source(name):
            path = port / name
            return path.read_text() if path.is_file() else None

        if not (port / "kernel.hip.cpp").is_file() and not vendor_bridge_layout(working_source):
            missing.append("kernel.hip.cpp")
        committed = all(prefix + name in committed_files for name in required) and (
            prefix + "kernel.hip.cpp" in committed_files or vendor_bridge_layout(
                lambda name: read_committed(posixpath.normpath(prefix + name))))
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
