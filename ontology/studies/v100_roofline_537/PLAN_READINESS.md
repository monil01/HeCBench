# Profile-plan readiness

`tools/v100_roofline_prepare.py` prepares source-bound per-model input/build plans. It **never builds, executes, profiles, or updates study records**. The compact `plan_audit.csv` and `plan_readiness_summary.json` describe all 537 × 4 assessment slots; full plans stay outside source Git and are reproducible.

## Regenerate

```bash
python3 tools/v100_roofline_prepare.py \
  --output-root /tmp/a-fresh-v100-roofline-plan-root
```

The default census is the hash-bound `source_census` in this study's protocol. `--source-root` supports a relocated source checkout and `--census` selects another explicit immutable snapshot. `--benchmark` is repeatable and `--models` selects models. Every selected model retains its own build, assets, dependency bindings, reference configuration, source qualification and effective timing-region metadata.

Output must be a fresh directory strictly under `/tmp`; existing directories are refused. No accepted output path is reused. Without `--stage`, only JSON plans are generated. With `--stage`, execution bindings are fully rehashed and explicitly declared runtime assets are copied into new scratch and made read-only. Default asset-copy budget is 64 MiB per model, with a 2 GiB free-space reserve. Builds, GPU execution and counters remain separate work.

## Meaning of readiness

- `input_plan_ready_ROI_and_kernel_mapping_still_required`: inputs and ordinary command structure have no detected staging gate. This is not a runtime sandbox guarantee or a profiling/classification result.
- `blocked`: an explicit gate needs review, for example a fixed capture directory, ambiguous external argument, custom build script, hardcoded source path or asset-copy budget.
- `native_hardware_prerequisite_unavailable`: the original native contract is unavailable; no substitute arithmetic or fabricated zero counters.

The five multi-device configurations cannot silently become GPU6-only contracts. Five native hardware barriers retain explicit null profile/classification evidence. Execution-only/runtime/transfer and integer-heavy cases are not forced onto a floating-point roofline.

SSS illustrates why isolation is mandatory: its selected command names an existing capture root. A new working directory and `STUDY_OUTPUT_DIR` alone do not redirect that root. Exact reviewed argv replacement or a separately source-bound profiling adapter is required before execution.

`--reviews` can apply exact result/config/dependency-bound isolation declarations and argv replacements. Output replacements must remain under the new scratch directory; read-only directory/file roles are explicit. Custom build scripts and literals require actual source review, not blind gate clearance. Original multi-device contracts, missing effective ROI and escaping/over-budget assets cannot be waived generically.

## Measurement remains separate

A profiler executor must rehash dependencies before/after build and execution under the shared GPU lock, pin actual executables/libraries, create a private TMPDIR, enforce storage/time bounds and preserve failed profiles. The planner does not run those checks or mutate measurement fields.

The root-validated inventory selector

```
--filter-mode per-launch-config --kernel-id '::regex:.*:1'
```

collects the first invocation of each observed function/geometry. It remains a **partial kernel sample** until setup, warmup, scientific stages, checker kernels, multiplicities and concurrency are source-mapped. Generic JACC names, vendor internal kernels and data-dependent paths need explicit review. Profile times do not replace the original five checked timing trials.

Measured FLOPs, useful scientific work and precision-compatible modeled/empirical ceilings remain distinct. Zero FP operations do not prove a memory bottleneck; mixed pipelines, unsupported counters, incomplete ROI and runtime-only work retain unknown or not-applicable classifications.

## Record ownership

Readiness publication updates only these fields inside each benchmark's `models[model]` record:

- `profile_plan_status`
- `profile_plan_blocking_gates`
- `profile_plan_configuration_id`

Existing `profiling_status`, measurement values, profile IDs, roofline/ontology classifications and all other fields remain unchanged. Readiness metadata therefore cannot promote a sample to a measured benchmark result.
