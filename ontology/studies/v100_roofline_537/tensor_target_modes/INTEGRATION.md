# Standalone loader integration

The pinned scientific manifest is manifest.json (SHA256 5be20948b0b10c49866406fa5ec70059cf8ce6a7439b70482c25af629f2198f5). Loader source and this interface note are separately bound by loader_installation.json; they are outside the scientific manifest to avoid a hash cycle. usage.json is the original tensor source-review effort. loader_usage.json will contain the separate implementation effort receipt.

Import tools/v100_roofline_tensor_modes.py using importlib.util, as existing standalone helpers do.

1. required_bindings("wmma", MODEL) returns the exact absolute-path -> SHA256 inventory. This is NOT approval. Under the caller's execution lock, independently rehash these files before/after the actual new profile and retain the values in that attempt provenance. The loader reads these files but does not acquire GPU/metadata locks, profile, compile or write anything.
2. For EACH parsed counter row call load_tensor_mode(benchmark="wmma", model=MODEL, kernel_name=g["identity"]["Kernel Name"], kernel_name_base="demangled", context=ctx, metrics=g["metrics"], captured_bindings=actual_attempt_hash_map). The name base refers to the actual printed row name, not necessarily the profiler selection name base. Do not pass a mangled row while labeling it demangled.
3. ctx requires configuration_id, reference_result_sha256, raw_config_sha256 and gpu_sm70=True. Any supplied architecture or observed instruction-mode information must agree. Missing or contradictory data produces approved=false, factor=null, precisionmode=null and a reason. Captured bindings must come from actual profile provenance; do not synthesize old capture bindings from current files.
4. Only approved=true permits passing tensor_flops_per_instruction and tensor_convention_evidence into the unchanged classifier's derive function for THAT ROW. Add returned tensor_precision_mode to a new row-specific context. Otherwise leave conversion disabled. No integration into collector/classifier has been performed.

The approved return also includes readonly_bindings, selected_target_SASS_observed=false, collector_default_changed=false and ROI_membership_approved=false. The physical512factor is scoped to FP16inputs/FP32accumulation/HMMA884/sm70. ScalarFP32epilogue stays separate. A known native geometry diagnostic shares the WMMA kernel family; precision-mode eligibility does not grant primaryROI membership.

CLI inventory: python3 tools/v100_roofline_tensor_modes.py --benchmark wmma --model ka --list-required
CLI decision: python3 tools/v100_roofline_tensor_modes.py --benchmark wmma --model ka --request REQUEST.json
REQUEST.json fields: kernel_name, kernel_name_base, context, metrics, captured_bindings. Decisions print JSON; exit0 approves, exit2 rejects. Inventory exit0 is explicitly approved=false.

The full collectively reviewed compiler/package receipt set is required for each model, deliberately conservative; Julia models additionally require their selected project/manifest/preferences. Exact paths are intentional; relocation or a new dependency revision needs a new source review/manifest rather than an unchecked path rewrite.

No implementation tests or GPU/build/runtime executions were performed. Source inspection was used. A decoded-regex audit separately resolved an earlier display-escaping false alarm; original scientific evidence was copied byte-for-byte and no capture was invalidated.
