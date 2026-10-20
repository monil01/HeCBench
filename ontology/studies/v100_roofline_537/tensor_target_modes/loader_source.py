#!/usr/bin/env python3
"""Read-only, fail-closed source/API tensor-mode approval for individual NCU rows.

Never profiles, compiles, writes metadata, or changes collector defaults.
Current file hashes AND actual profile provenance bindings must agree.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Mapping

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUNDLE = ROOT / "ontology/studies/v100_roofline_537/tensor_target_modes"
MANIFEST_SHA256 = "5be20948b0b10c49866406fa5ec70059cf8ce6a7439b70482c25af629f2198f5"
METRIC = "sm__inst_executed_pipe_tensor.sum"
PRECISION_MODE = "FP16inputs_FP32accumulation_HMMA884_sm70"
MODELS = frozenset(("cuda", "cuda_jl", "ka", "jacc"))


class ApprovalError(ValueError):
    """Evidence is missing, changed, ambiguous, or outside reviewed scope."""


def _digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _bundle(bundle_dir):
    directory = Path(bundle_dir) if bundle_dir is not None else DEFAULT_BUNDLE
    try:
        raw = (directory / "manifest.json").read_bytes()
        if hashlib.sha256(raw).hexdigest() != MANIFEST_SHA256:
            raise ApprovalError("Unrecognized or changed approval manifest")
        manifest = json.loads(raw)
        for filename, expected in manifest["files"].items():
            relative = Path(filename)
            if relative.is_absolute() or ".." in relative.parts:
                raise ApprovalError("Unsafe bundle relative path")
            if _digest(directory / relative) != expected:
                raise ApprovalError("Changed archived evidence: " + filename)
        review = json.loads((directory / "review.json").read_bytes())
        table = json.loads((directory / "approval_table.json").read_bytes())
        if table["approvals"] != review["approval_table"]:
            raise ApprovalError("Approval table disagrees with source review")
        if manifest["schema_version"] != 1 or review["schema_version"] != 1:
            raise ApprovalError("Unsupported evidence schema")
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ApprovalError("Cannot read complete evidence: " + str(exc)) from exc
    return review, table, directory


def _selected(review, benchmark, model):
    if benchmark != "wmma" or model not in MODELS:
        raise ApprovalError("No reviewed conversion for requested benchmark/model")
    rows = [r for r in review["wmma_records"]
            if r["benchmark"] == benchmark and r["model"] == model]
    if len(rows) != 1:
        raise ApprovalError("Missing or ambiguous selected source record")
    record = rows[0]
    if record["actual_selected_SASS_observed"] is not False:
        raise ApprovalError("Unexpected selected SASS qualification")
    mode = record["source_bound_mode"]
    expected = {"input_A": "IEEEFloat16", "input_B": "IEEEFloat16",
                "accumulator": "Float32", "tensor_result": "Float32",
                "C": "Float32", "D": "Float32"}
    if any(mode[k] != v for k, v in expected.items()) or mode["fragment_shape"] != [16, 16, 16]:
        raise ApprovalError("Source mode outside calibrated convention")
    return record


def _bindings(review, record):
    """Exact paths are intentional; relocated inputs need a new review."""
    result = {}

    def add(item):
        name, expected = item["path"], item["sha256"]
        if not isinstance(name, str) or not Path(name).is_absolute():
            raise ApprovalError("Binding must be an absolute file path")
        if not isinstance(expected, str) or re.fullmatch(r"[0-9a-f]{64}", expected) is None:
            raise ApprovalError("Invalid SHA256 binding")
        if name in result and result[name] != expected:
            raise ApprovalError("Conflicting binding: " + name)
        result[name] = expected

    add(record["reference_result"])
    add(record["raw_config"])
    for item in record["execution_bindings"]:
        if item["matches_review"] is not True:
            raise ApprovalError("Unmatched source review binding")
        add(item)
    # Deliberately require the complete collectively reviewed dependency bundle.
    for item in review["current_dependency_receipts"]:
        add(item)
    environment = record["dependency_environment"]
    if environment is not None:
        for item in review["environment_manifests"][environment]["files"]:
            add(item)
    add(review["calibration"])
    return dict(sorted(result.items()))


def required_bindings(benchmark, model, *, bundle_dir=None):
    """Return expected read-only bindings to capture before/after profiling.

    Inventory only, never approval. Caller owns locking and independent
    capture of these hashes in the actual profile provenance.
    """
    review, _, _ = _bundle(bundle_dir)
    return _bindings(review, _selected(review, benchmark, model))


def load_tensor_mode(*, benchmark, model, kernel_name, kernel_name_base,
                     context, metrics, captured_bindings, bundle_dir=None):
    """Return per-row conversion, or unknown with no factor.

    Context requires configuration_id, reference_result_sha256,
    raw_config_sha256, gpu_sm70=True. Optional architecture fields must agree.
    Supply the actual demangled counter-row name and parser numeric metric map.
    captured_bindings is an absolute-path -> SHA256 map from profile provenance.
    """
    denied = {"approved": False, "status": "unknown",
              "tensor_flops_per_instruction": None, "tensor_precision_mode": None,
              "tensor_convention_evidence": None,
              "selected_target_SASS_observed": False,
              "collector_default_changed": False}
    try:
        review, table, directory = _bundle(bundle_dir)
        record = _selected(review, benchmark, model)
        if not all(isinstance(x, Mapping) for x in (context, metrics, captured_bindings)):
            raise ApprovalError("Context, metrics and actual profile bindings must be mappings")
        if context.get("gpu_sm70") is not True:
            raise ApprovalError("Explicit gpu_sm70=True context required")
        for field, allowed in (("gpu_architecture", ("sm_70",)),
                               ("architecture", ("sm_70",)),
                               ("compute_capability", ("7.0", "sm_70")),
                               ("gpu_compute_capability", ("7.0", "sm_70"))):
            if field in context and context[field] not in allowed:
                raise ApprovalError("Contradictory architecture: " + field)
        for field, expected in (("device__attribute_compute_capability_major", 7),
                                ("device__attribute_compute_capability_minor", 0)):
            if field in metrics and metrics[field] != expected:
                raise ApprovalError("Contradictory measured architecture: " + field)
        for field, expected in (
            ("configuration_id", record["configuration_id"]),
            ("reference_result_sha256", record["reference_result"]["sha256"]),
            ("raw_config_sha256", record["raw_config"]["sha256"]),
        ):
            if context.get(field) != expected:
                raise ApprovalError("Selected configuration/result mismatch: " + field)
        for field, expected in (("benchmark", benchmark), ("model", model)):
            if field in context and context[field] != expected:
                raise ApprovalError("Contradictory context: " + field)
        if kernel_name_base != "demangled" or not isinstance(kernel_name, str):
            raise ApprovalError("Actual demangled kernel-row name required")
        value = metrics.get(METRIC)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ApprovalError("Measured numeric tensor instruction metric missing")
        if not math.isfinite(value) or value < 0:
            raise ApprovalError("Tensor metric must be finite and nonnegative")
        observed = context.get("observed_tensor_precision_mode")
        if observed is not None and observed != PRECISION_MODE:
            raise ApprovalError("Observed target mode contradicts reviewed API mode")
        opcodes = context.get("observed_tensor_SASS_opcodes")
        if opcodes is not None and (
            not isinstance(opcodes, list) or not opcodes or any(
                not isinstance(op, str)
                or re.fullmatch(r"HMMA\.884\.F32\.F32\.STEP[0-3]", op) is None
                for op in opcodes)):
            raise ApprovalError("Observed SASS outside reviewed tensor mode")
        matches = []
        for item in table["approvals"]:
            if item["benchmark"] != benchmark or item["model"] != model:
                continue
            if item["kernel_name_base"] != "demangled" or not item["kernel_regex"].startswith("^"):
                raise ApprovalError("Unanchored or unexpected approval name base")
            if re.search(item["kernel_regex"], kernel_name):
                matches.append(item)
        if len(matches) != 1:
            raise ApprovalError("No unique exact reviewed kernel family match")
        approval = matches[0]
        if (approval["decision"] != "approved_mode_for_conditional_calibrated_tensor_projection"
                or approval["architecture"] != "sm_70_only"
                or approval["metric"] != METRIC
                or approval["flops_per_instruction"] != 512
                or approval["selected_target_SASS_observed"] is not False):
            raise ApprovalError("Approval outside reviewed convention")
        bindings = _bindings(review, record)
        for name, expected in bindings.items():
            if captured_bindings.get(name) != expected:
                raise ApprovalError("Missing/changed actual profile binding: " + name)
            if _digest(Path(name)) != expected:
                raise ApprovalError("Current bound input changed: " + name)
        calibration = json.loads(Path(review["calibration"]["path"]).read_bytes())
        convention = calibration["conventions"]["tensor"]
        if (calibration["architecture"] != "sm_70"
                or convention["flops_per_sm_inst_executed_pipe_tensor"] != 512):
            raise ApprovalError("Calibration mode/factor changed")
        return {
            "approved": True, "status": "source_API_mode_approved_per_kernel",
            "tensor_flops_per_instruction": 512,
            "tensor_precision_mode": PRECISION_MODE,
            "tensor_convention_evidence": {
                "approval_manifest": str(directory / "manifest.json"),
                "approval_manifest_sha256": MANIFEST_SHA256,
                "review": str(directory / "review.json"),
                "review_sha256": _digest(directory / "review.json"),
                "calibration": review["calibration"],
                "benchmark": benchmark, "model": model,
                "configuration_id": record["configuration_id"],
                "implementation": approval["implementation"],
                "kernel_name": kernel_name, "kernel_name_base": "demangled",
                "kernel_regex": approval["kernel_regex"], "basis": approval["basis"],
                "dependency_qualification": record["dependency_qualification"],
                "selected_target_SASS_observed": False,
                "retained_diagnostic_probe_is_selected_binary": False,
                "dependency_receipts_are_prospective": True,
            },
            "readonly_bindings": bindings,
            "selected_target_SASS_observed": False,
            "collector_default_changed": False,
            "ROI_membership_approved": False,
            "scope": "Physical tensor work; scalar epilogue separate; no ROI, portability or causal approval.",
        }
    except (ApprovalError, OSError, KeyError, TypeError, ValueError, OverflowError, re.error) as exc:
        return {**denied, "reasons": [str(exc)]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--bundle-dir", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--list-required", action="store_true")
    parser.add_argument("--request", type=Path, help="JSON kernel/context/metrics/captured_bindings")
    args = parser.parse_args()
    if args.list_required:
        try:
            answer = {"readonly_bindings": required_bindings(
                args.benchmark, args.model, bundle_dir=args.bundle_dir),
                "approved": False, "scope": "Inventory only"}
        except ApprovalError as exc:
            answer = {"approved": False, "status": "unknown", "reasons": [str(exc)]}
        print(json.dumps(answer, indent=2))
        return 0 if "readonly_bindings" in answer else 2
    if args.request is None:
        parser.error("--request or --list-required required")
    request = json.loads(args.request.read_bytes())
    answer = load_tensor_mode(
        benchmark=args.benchmark, model=args.model,
        kernel_name=request.get("kernel_name"),
        kernel_name_base=request.get("kernel_name_base"),
        context=request.get("context"), metrics=request.get("metrics"),
        captured_bindings=request.get("captured_bindings"), bundle_dir=args.bundle_dir)
    print(json.dumps(answer, indent=2))
    return 0 if answer["approved"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
