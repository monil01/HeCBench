# Root-reviewed Nsight2025 method-calibration controller

**Prepared only. Author did not run the controller, profiler, canaries, GPU queries,
compiler/builds or tests.** Only source reading, file hashing, static AST inspection
and reading old recorded counters occurred. No active source tool, ledger, paper,
primary measurement or method-adoption record was changed.

## Exact source and launch

`delivery.json` freezes controller/manifest SHA and the exact `root_launch_argv`.
Use its argv directly, including **absolute roots**, system `/usr/bin/python3`,
manifest SHA and explicit `--execute`. Suggested fresh output is
`raw/method_calibration_ncu2025_2_0_001`; the script rejects existing/symlink outputs,
outputs outside the owned raw tree, and batch* names that could be mistaken for
benchmark profiler evidence. No output directories exist yet. Without --execute,
only argument/source-manifest validation occurs, no process is run or output made.

Root should include this controller's actual PID/start ticks plus exact NUL-delimited
command SHA in its custom GPU-controller review before a new supervisor waits for
it. Do not add CPU refresh to that controller frontier. The calibration command
waits for the same GPU lock itself; **no outerflock**. No active controller is stopped.

The source manifest binds83 files: collector32a9cf...exact source, current parser,
Landlock guard, system interpreter/NVML executable, installed2025wrapper+ELF+Linux
injection/NvPerf/launcher files+sections, both immutable existing canary binaries,
full .cu sources, old raw counters/logs/provenance, original calibration review and
preparation/controller source. Manifest SHA is a CLI prerequisite. Every binding is
rechecked before, after each case and at final completion. Source/tool drift fails
closed; root must preserve/explicitly revise method strata rather than update hashes
in place. Original old compiler executable identity is unknown: only its build argv
was captured. Reuse exact existing ELF bytes; no rebuild or retrospective compiler
hash claim is made.

## Execution and retention policy

1. Import exact root collector **32a9cfeba983dc3ae8e5d9b8bb85ca4e7736821be0a9c206bb57b80c7471da4c**,
   reusing its run, wait_idle_snapshot, own-session quiescence, foreign-GPU monitor,
   deadline/reserve handling and child write guard. This does not call collect() or
   register a benchmark, execute builds, summarize or adopt a method.
2. Hold existing `/tmp/hecbench-julia-study-gpu.lock` across both calibrations. Record
   lock queue wait separately. Each per-case idle wait is unbounded and recorded
   separately from the operational process wall deadline. Foreign processes are
   never signaled; imported runner may stop only its own fresh session on the
   existing reviewed timeout/foreign/reserve policy.
3. Create fresh private TMPDIR/CUDA_CACHE_PATH/XDG_CACHE_HOME in each case. Only the
   new output tree is writable under Landlock; no global or existing caches need
   writes. Runtime PATH=/usr/bin:/bin and empty LD_LIBRARY_PATH/LD_PRELOAD are an
   explicit new calibration environment, avoiding author-session PATH ephemera.
   Other inherited variables retain ordinary values and primary binaries are not
   changed. UUID-only CUDA_VISIBLE_DEVICES is fixed to physicalGPU6's known UUID.
4. Check GPU6 UUID/BDF via guarded, monitored NVML CLI; parse every raw counter row's
   PCI domain/bus/device and compute capability7.0. Installed NCU runtime --version
   must contain Version2025.2.0.0. All source checks and process logs persist.
5. Request exact **full collector.METRICS** with target-processesall, mangled
   per-launch-config firstfunction/shape selector, cache-controlall, clock-controlnone,
   kernelreplay and base raw CSV units. No metric is dropped on an error. Require
   every metric in every row to be finite/nonnegative with explicit expected units,
   positive duration, exact identity set and no duplicates.
6. Reuse existing precision_calibration/canary and half_arithmetic_calibration/canary;
   no compile or target instrumentation source change. All actual original full
   array host checks must complete with the exact bound calibration_correct=true
   line once per program. These original checks inspect both half2 lanes, scalar
   half, all4096Float32/Float64 outputs and all256WMMA outputs; arithmetic source
   checks all4096 packed ADD/MUL lanes. Process rc0 alone is insufficient.
7. Retain complete counters/execution/identity/version logs, actual command/env,
   source before/after hashes, all raw process outcomes and copied controller/
   collector/parser sources even on failure. No scratch cleanup occurs. Failure
   status does not adopt new conventions or trigger benchmark recapture.

## Nine rows and physical versus useful work

| Source family/shape | Exact reported instruction count | Physical executed FLOPs | Useful source FLOPs |
|---|---:|---:|---:|
| half2HFMA,128threads×32blocks |524288|2097152|2097152|
| scalarhalfHFMA,128×32 |524288|2097152|1048576|
| HMMA884 WMMA,32×1 |2048|1048576|1048576|
| FP32FFMA,128×32 |4096|8192|8192|
| FP64DFMA,128×32 |4096|8192|8192|
| FP32FFMA,256×16 |4096|8192|8192|
| FP64DFMA,256×16 |4096|8192|8192|
| packedHADD2,128×32 |524288|1048576|1048576|
| packedHMUL2,128×32 |524288|1048576|1048576|

The original review's **physical packed instruction convention remains unchanged**:
HFMA2 has2lanes×2ops even when scalar source uses only one lane. The scalar useful
column must not replace that physical conversion. ADD/MUL2 have2physicalFLOPs.
Tensor factor512 is specific to VoltaHMMA884 FP16inputs/FP32accumulation.

The new raw typed instruction and architectural instruction-peak counts/units
must equal old recorded invariant counters. New duration/clocks/traffic are **not**
required to equal old observations and never replace primary performance means.
Old/new full requested metric availability is explicitly checked; older references
only requested a subset, so newly requested metrics are not falsely called old-
measured invariants. Unknown/missing/unsupported data, counter mismatches, bad full
host outputs, wrong PCI, source drift or process stops all fail closed.

## Meaning of a successful result

`new_profiler_calibration_pass_root_adoption_still_required` establishes only these
nine source-known instruction/shape observations with existing binaries on V100.
It does not validate the separate miniDGS/host-library issue, every replay contract,
other precision/architectures, benchmark ROI coverage, scientific correctness,
causal roofline classification or equivalence of old/new timing/traffic. Root must
review successful receipts and decide any new tool-version adoption/benchmark
recapture separately. The original2022counter/correctness evidence remains immutable.

Preparation shared usage is separately captured/closed in usage_receipt.json, not
assigned to benchmark translation attempts, deviceJIT checks or per-model costs.
