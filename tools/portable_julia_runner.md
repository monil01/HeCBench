# Four-model study runner

`python3 tools/verify_portable_julia.py --config <configuration.json>`

The runner discovers legacy or nested sources (`--discover BENCH`), but requires
an explicit per-benchmark configuration because shared inputs, native verifiers,
and timing equivalence cannot be inferred safely. Keep source-native defaults
separate from deterministic study configurations. All commands are argument
arrays, not shell strings. All builds and executions run from isolated temporary working directories.
Relative .jl/.cu source arguments resolve to original absolute source paths;
original source files remain read-only and binary outputs stay temporary.
Commands receive STUDY_OUTPUT_DIR. Optional runtime_assets lists fixture paths
relative to the model source cwd to copy into the isolated working directory. `{arch}` in a build argument is replaced using the selected GPU's actual
compute capability. Includes relative to the original C/CUDA source file work because the source
argument resolves to its original absolute path. Julia @__DIR__ similarly
identifies the original source directory. Write outputs to pwd() or
ENV["STUDY_OUTPUT_DIR"], never the source directory.

```json
{
  "benchmark": "BENCH",
  "configuration_id": "deterministic_n1024_v1",
  "gpu": 6,
  "timeout": 300,
  "inputs_shared_verified": true,
  "timed_regions_equivalent_verified": true,
  "input_contract": {"precision":"float32","size":1024,"fixture":"describe exact values, reset and seed"},
  "atol": 0.000001,
  "rtol": 0.00001,
  "models": {
    "cuda": {
      "cwd":"src/BENCH-julia",
      "build":["nvcc","-O3","-arch={arch}","study_native.cu","-o","study_native"],
      "command":["./study_native"],
      "structured":true,
      "verification_method":"all_elements_cpu_reference",
      "native_check_description":"Describe CPU formula and tolerances"
    },
    "cuda_jl": {
      "command":["julia","--project=/home/imo/HeCBench_julia/src/_julia_ka_env","study.jl"],
      "structured":true,
      "verification_method":"all_elements_cpu_reference"
    },
    "ka": {
      "command":["julia","--project=/home/imo/HeCBench_julia/src/_julia_ka_env","main.jl"],
      "structured":true,
      "verification_method":"all_elements_cpu_reference"
    },
    "jacc": {
      "command":["julia","--project=/home/imo/HeCBench_julia/src/_julia_jacc_env","main.jl"],
      "structured":true,
      "verification_method":"all_elements_cpu_reference"
    }
  }
}
```

Each structured command must emit exactly five lines beginning `STUDY_TRIAL `,
followed by JSON with `trial` (1--5), `time_seconds`, `correct`, `checked_values`,
`max_abs_error`, `max_rel_error`, `timing_region`, `jit_excluded`, and
`internal_repeats`. Use explicit device warmup before trial 1, synchronization
at both timing boundaries, and input reset outside kernel-only timing. The
runner records five timings and averages only 2--5; correctness applies to all.
`values` optionally contains actual numeric output summaries for cross-checks.

For direct output comparison, add `output_artifact` with `path` (relative to
command cwd or absolute), `dtype` (`float32`, `float64`, `int32`, `int64`,
`uint32`, `uint64`), `count`, and `byte_order:"little"`. Write a distinct raw
binary file per trial. Outputs are copied into the run bundle and hashed before
native temporary files are removed. Nonfinite values and count/shape mismatch
are rejected. Native-only checks are labeled `independent_native_checks_only`,
not direct CUDA/Julia equivalence. Numerical summary comparisons retain their
weaker evidence label. Shared input claims and verifier descriptions remain
benchmark-specific audited evidence, not guarantees supplied by this runner.

Legacy command adapters run five separate processes and require explicit
`time_regex`, `time_scale_seconds`, `jit_excluded`, `timing_region`,
`internal_repeats`, `native_check_audited`, `pass_regex`, `checked_values`, and
optional `value_regexes`. A cold Julia timer cannot set `jit_excluded:true`.
Avoid legacy adapters when study harnesses can produce structured output.

Run bundles live under `.porting-state/portable_julia/runs/<run_id>/`; SQLite is
`.porting-state/portable_julia/results.db`. Compiler/build and execution/device
JIT invocations are captured as events. These events are not automatically
candidate translation-attempt counts: supply candidate/repair counts and token
provenance in `attempt_accounting` and `token_accounting`; unknown stays null.
Preserve infrastructure effort separately from benchmark translation.

The runner locks `/tmp/hecbench-julia-study-gpu.lock` and sets
`CUDA_VISIBLE_DEVICES=6` by default. Do not wrap the runner itself in that same
lock (nested lock deadlock). GPU worker commands launched outside the runner
must acquire the lock themselves.

Internal model keys cuda/cuda_jl/ka/jacc have explicit exported mapping to
the canonical study IDs cuda/julia/julia-ka/julia-jacc. The result JSON records
both this mapping and source/adapter hashes. Variant verification is independent
for KA-first operation; a pending JACC does not invalidate KA. Any observed
CUDA-vs-CUDA.jl disagreement blocks variant verification. Exit zero requires
both-reference verification for each requested variant, at the recorded evidence
strength. A four-model matched timing flag additionally requires all numerical
self-checks, shared inputs, equivalent regions and JIT-excluded positive times.

## Reuse without another five-run series

For KA-first production, preserve the accepted native/CUDA.jl/KA trial bundle.
The JACC configuration can add `reuse_result` (the absolute result.json path)
and `reuse_models:["cuda","cuda_jl","ka"]`; only new JACC trials execute.
The runner checks benchmark, protocol, input contract, tolerances, GPU,
configuration ID, model command/configuration, source hashes, completeness, and
individual adapter/dependency file hashes. Add each model's support-code paths
to `source_dependencies` before the initial KA run. Reused rows identify their
original run ID, and their artifact paths continue to reference that bundle.
The original bundle is therefore a required durable dependency of the result.

A controlled resource ablation may reuse unchanged reference/KA measurements
across named configurations by setting `allow_cross_configuration_reuse:true`
and a nonempty `cross_configuration_reuse_reason`. Only the configuration ID
check is relaxed. Benchmark, protocol, fixture, tolerances, GPU, shared-input
and equivalent-region declarations, exact model specifications, source hashes,
and dependency hashes must remain unchanged. Results and reuse events record
both IDs, the reason, and the unchanged models. For example, changing only a
JACC launch's shared-memory allocation can reuse native/CUDA.jl/KA evidence;
the changed JACC source must execute a new five-trial series. Preserve the old
configuration as the default-resource result, and analyze the new controlled
configuration separately.

Early bundles created before individual adapter hashes existed require an
explicit manual unchanged-adapter audit, recorded with
`reuse_legacy_adapter_unchanged_verified:true`. This is a disclosed migration
exception, not a general bypass for changed source/configurations. A source,
fixture, runtime/compiler environment, checker, or timing correction requires
new appropriately labeled measurements; retain the superseded evidence.

`--export` refreshes aggregate `timings.csv` from all saved SQLite rows with
configuration IDs, canonical model IDs, raw trials, means, numerical/matched flags, and reuse IDs.
It runs automatically after new completed bundles and uses an atomic file
replacement under an export lock. Reused rows are explicitly tagged: analysis
must select one final matched bundle per configuration and must not count the
same source trial set twice. Original authoritative SQLite remains `results.db`
for compatibility with running workers; `verification.db` is a relative
symlink to it. Resolve that symlink before writes through external tools.
New results record the verifier's SHA256. Earlier bundles precede that field;
use their source commit and the saved execution event provenance instead of
inventing a historical tool hash.

Raw artifact decoding additionally supports `uint8`, `int8`, `uint16`, and
`int16` (explicit little endian), allowing image/pixel storage to remain exact.
New bundles archive verifier_source.py and hash bytes captured at process
startup, so later concurrent tooling edits cannot change the recorded hash of
the running verifier. Existing result bundles retain their original provenance.

Identical SHA256-verified artifact bytes use a full finite-value scan once per
unique floating-point content/dtype/count, then reuse that validation across
identical repeated outputs. Both artifact byte sizes/counts are checked. Integer
storage is finite by construction. Nonidentical bytes use the numerical
absolute/relative comparison, preserving signed-zero and tolerance behavior.
This avoids repeated large Python allocations without weakening checks.

## Source stability while queued

The runner hashes model source and explicitly declared adapter dependencies
before launching. It checks those exact hashes before/after native compilation
and, after acquiring the GPU lock, before/after each execution. Changes reject
the affected model as `provenance_invalid`; its logs and mutation event remain,
but its trial rows cannot establish numerical verification or matched timing.
Do not edit a benchmark or its dependencies until its entire runner process
has completed. This detects a mutation even when the new file happens to pass
numerical checks. It does not infer unlisted dependencies: declare every
included support file in `source_dependencies`.

For new configurations set `bind_all_declared_dependencies: true` per model.
Every declared `source_dependencies` and `runtime_assets` file is then required
and hashed before execution and at stability/reuse checks, including `.inc`,
`.bin`, and `.txt` files. Legacy default bindings are preserved. Do not switch
this option on in a reused model spec and imply historical hashes existed.
