#!/usr/bin/env python3
"""Native WMMA path-only profiling plan. CPU source staging only; never builds/runs."""
import argparse, copy, difflib, hashlib, json, os, pathlib
P = pathlib.Path
OLD = "/home/imo/HeCBench_julia/.porting-state/portable_julia/diagnostics/wmma"
EXPECTED = "cb301577b4939da78a9603b13af59914dd78d91032741c3c17c2138f1391f386"
GATE = "source_bound_absolute_path_literals_require_write_isolation_review"
COMPILER = "/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda/12.9/bin/nvcc"


def sha(p):
    h = hashlib.sha256()
    with P(p).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def dump(p, obj):
    with P(p).open("x") as f:
        json.dump(obj, f, indent=2)
        f.write("\n")


def main():
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument("--base-plan", type=P, required=True)
    a.add_argument("--output-root", type=P, required=True)
    a.add_argument("--source-root", type=P, default=P("/home/imo/HeCBench_julia"))
    a.add_argument("--stage", action="store_true", help="Copy text/declared input assets; no build/run")
    z = a.parse_args()
    root, out = z.source_root.resolve(), z.output_root.absolute()
    if out.exists() or out.is_symlink() or not out.resolve().is_relative_to(P("/tmp")) or out.resolve() == P("/tmp"):
        a.error("Output must be fresh strictly under /tmp")
    base = json.loads(z.base_plan.read_text())
    if base["benchmark"] != "wmma" or base["model"] != "cuda":
        a.error("Only selected native WMMA plan is reviewed")
    if set(base["blocking_gates"]) != {GATE}:
        a.error("Refuse clearing unrelated or missing blocking gates")
    for name, expected in base["execution_bindings"].items():
        if sha(name) != expected:
            a.error("Selected input drift: " + name)
    rp = P(base["reference_result"])
    if sha(rp) != base["reference_result_sha256"] or sha(rp.with_name("config.json")) != base["raw_config_sha256"]:
        a.error("Selected result/config drift")
    original = root / "src/wmma-julia/study_native.cu"
    if sha(original) != EXPECTED or base["execution_bindings"].get(str(original)) != EXPECTED:
        a.error("Unreviewed native adapter")
    command = base["build_command"]
    if command[0] != COMPILER or command.count(str(original)) != 1 or "-arch=sm_70" not in command:
        a.error("Require exact recorded CUDA12.9 compiler/source/sm70")
    if base["environment_overrides"]["CUDA_VISIBLE_DEVICES"] != "6":
        a.error("Selected GPU visibility contract changed")
    slot = out / "wmma/cuda"
    scratch = slot / "scratch"
    overlay = scratch / "profile_overlay_src"
    diagnostic = scratch / "profile_outputs/wmma_geometry"
    target_source = overlay / "wmma-julia/study_native.cu"
    text = original.read_text()
    before = 'const char*base="' + OLD + '";'
    after = 'const char*base=' + json.dumps(str(diagnostic)) + ';'
    if text.count(before) != 1:
        a.error("Exact single diagnostic path anchor missing")
    changed = text.replace(before, after, 1)
    # Inverse transformation proves every byte outside one host path is unchanged.
    if changed.replace(after, before, 1) != text:
        a.error("Path-only inverse reconstruction failed")
    out.mkdir(parents=True)
    slot.mkdir(parents=True)
    scratch.mkdir()
    (scratch / "profile_outputs").mkdir()
    (scratch / "ncu_tmp").mkdir()
    diagnostic.mkdir()
    sources = {
        target_source: changed.encode(),
        overlay / "wmma-julia/study_cases.h": (root / "src/wmma-julia/study_cases.h").read_bytes(),
        overlay / "wmma-cuda/main.cu": (root / "src/wmma-cuda/main.cu").read_bytes(),
        overlay / "wmma-cuda/reference.h": (root / "src/wmma-cuda/reference.h").read_bytes(),
    }
    patches = []
    for dest, data in sources.items():
        patches.append({"destination": str(dest), "sha256": hashlib.sha256(data).hexdigest(),
                        "bytes": len(data), "staged": z.stage})
        if z.stage:
            dest.parent.mkdir(parents=True, exist_ok=True)
            with dest.open("xb") as f:
                f.write(data)
    diff = "".join(difflib.unified_diff(text.splitlines(True), changed.splitlines(True),
                                       fromfile=str(original), tofile=str(target_source)))
    with (slot / "native_adapter_path_only.diff").open("x") as f:
        f.write(diff)
    old_scratch = base["scratch_cwd"]
    def move(s):
        return s.replace(old_scratch, str(scratch)) if isinstance(s, str) else s
    plan = copy.deepcopy(base)
    plan["scratch_cwd"] = str(scratch)
    plan["build_command"] = [str(target_source) if x == str(original) else move(x) for x in command]
    plan["profile_target_command"] = [move(x) for x in base["profile_target_command"]]
    plan["environment_overrides"] = {k: move(v) for k, v in base["environment_overrides"].items()}
    plan["environment_overrides"].update(STUDY_OUTPUT_DIR=str(scratch / "profile_outputs"),
                                          TMPDIR=str(scratch / "ncu_tmp"))
    for asset in plan["runtime_assets"]:
        asset["destination"] = move(asset["destination"])
        dest = P(asset["destination"])
        if not dest.resolve().is_relative_to(scratch.resolve()):
            raise ValueError("Runtime asset escapes new scratch")
        if z.stage:
            if not P(asset["source"]).is_file():
                raise ValueError("Only declared WMMA fixture files supported")
            for receipt in asset["source_files"]:
                if sha(receipt["path"]) != receipt["sha256"]:
                    raise ValueError("Runtime fixture drift")
            with dest.open("xb") as f:
                f.write(P(asset["source"]).read_bytes())
    helper = P(__file__).resolve()
    extra = {str(helper): sha(helper), str(z.base_plan.resolve()): sha(z.base_plan),
             str(original): EXPECTED, COMPILER: sha(COMPILER)}
    if z.stage:
        extra.update({str(p): sha(p) for p in sources})
        plan["execution_bindings"].update({str(p): sha(p) for p in sources})
    entry = {
        "id": "wmma_native_diagnostic_output_only_private_source_overlay_v1",
        "source_hash": EXPECTED, "source_path": str(original),
        "additional_current_readonly_input_bindings": extra,
        "path_roles": [
            {"literal": OLD, "role": "write-only excluded geometry diagnostic directory",
             "outputs": ["impl1_96x80_output_f32.bin", "native_geometry.json"],
             "replacement": str(diagnostic), "source_line": 20},
            {"literal": "input_u32.bin", "role": "relative fixture read in scratch; bytes unchanged"},
            {"literal": "oracle_f32.bin", "role": "relative oracle read in scratch; bytes unchanged"},
            {"literal": "STUDY_OUTPUT_DIR", "role": "private trial outputs, unchanged application behavior"},
        ],
        "path_only_change": {"old_anchor": before, "new_anchor": after,
                             "inverse_reconstructs_original_bytes": True,
                             "diff_path": str(slot / "native_adapter_path_only.diff"),
                             "diff_sha256": sha(slot / "native_adapter_path_only.diff")},
        "original_kernel_source_copies": patches,
        "compiler": COMPILER, "compiler_sha256": sha(COMPILER),
        "math_or_device_body_changed": False,
        "diagnostic_execution_retained": True,
        "diagnostic_in_primary_ROI": False,
        "primary_correctness_or_timing_recertified": False,
        "tensor_mode_qualification": "Source/API mode unchanged by sole host path literal. Fresh collector must independently bind compiled overlay, original source, dependencies and binary; no target SASS observed by this staging helper.",
        "write_guard_required": "Existing fresh-attempt Landlock guard; never allow writes to old diagnostic directory",
        "GPU_build_or_target_executed_by_helper": False,
    }
    dump(slot / "isolation_review.json", entry)
    plan["review"] = {"path": str(slot / "isolation_review.json"),
                      "sha256": sha(slot / "isolation_review.json"), "entry": entry}
    plan["blocking_gates"] = []
    plan["staged"] = z.stage
    plan["staging_status"] = ("input_plan_ready_native_WMMA_path_only_overlay_ROI_unreviewed"
                              if z.stage else "CPU_proposal_only_not_staged")
    plan["commands_executed"] = False
    plan["source_path_literal_findings"] = base["source_path_literal_findings"]
    plan["notes"] += ["Original absolute write path retained as audited finding, redirected only in private copied adapter. Runtime/audit/geometry diagnostic executes unchanged. Primary accepted files remain read-only."]
    dump(slot / "plan.json", plan)
    dump(out / "summary.json", {"source_bound_review": str(slot / "isolation_review.json"),
         "plan": str(slot / "plan.json"), "stage_only": z.stage,
         "build_or_target_executed": False,
         "GPU_or_primary_metadata_modified": False})
    print(json.dumps({"plan": str(slot / "plan.json"), "staged": z.stage,
                      "build_or_target_executed": False}))


if __name__ == "__main__":
    main()
