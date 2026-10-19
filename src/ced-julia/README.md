# CED: source Canny-derived GPU pipeline

## Certified scope

`ced_defined_zero_borders_12resident_frames_full4GPUstages_3sizes_aligned_H0_v2`
contains all four original GPU stages: Gaussian smoothing, Sobel magnitude and
orientation, non-maximum suppression, and the original point threshold. Twelve
resident byte images cover three shapes (34×50,66×98,130×194) and four patterns.
The source GPU bodies are unchanged in the native adapter. CUDA.jl, KA and JACC
implement actual device kernels; each passed its first candidate/device-JIT check.
All three are `framework_device_kernel`, with independently reviewed mappings.

This is a **conditional native reference**. Actual original parser ASAN reports a
four-byte write into a one-byte destination; source intermediate borders produce
356 uninitialized-read initcheck errors. The named checked binary loader and
zero-defined intermediate/angle borders repair those issues. The repaired full
four-stage pipeline reports zero initcheck errors. Original source, exact host
patch, extraction proof and actual diagnostic hashes are retained. The external
DVC frame dataset and original dataset CLI are unverified. CPU/mixed/dynamic
partitioning and host-thread orchestration are outside this study.

## Numerical evidence

Final receipt: `.porting-state/portable_julia/runs/20261019T190613247303Z_ced/result.json`.
All20 primary captures are bitidentical, each containing803,730 Int32-encoded
values, including401,922 actual input/coefficient/guard prefix values. Complete
surviving output/intermediate/orientation buffers and guards are captured after
every timed trial. All four versions additionally execute one **untimed** complete
stage trace (669,680 values) after the five primary trials. Those are actual stage
outputs, not a claim to have recovered overwritten timed intermediates.

The independent Python coordinate oracle uses integer weighted floors for the
source Gaussian, separate Sobel derivative/norm/direction expressions, directional
NMS and point-threshold predicates. Cross-GPU tolerances are zero. CPU budgets were
declared before execution: exact Gaussian/angles/edges/guards, absolute1 for Sobel
and NMS magnitudes; all observed errors are zero. Actual coefficient values/bits
are read before and after execution. JACC's auxiliary324-element template is
actually read back before/after (648 zeros); a coordinate coverage proof checks
that all324 shared values are overwritten exactly once before arithmetic reads.

The source Gaussian truncates *each* contribution, and the source final stage
uses a point threshold40 rather than connectivity hysteresis. Both are preserved
method choices, not automatically scientific defects. A source-GPU component
witness and independent8-neighbor connectivity control differ at two pixels.
No canonical Canny/application-quality certificate is claimed.

## Timing and mapping

Five actual in-process trials per version, exact GPU/host warmup outside timing,
discard only trial1 and average2–5. ROI:12 resident images ×4 stages =48 launches
plus whole-device completion. Allocation, transfers, output checks, original
host threads/partitioning and IO are excluded symmetrically. Native chrono and
Julia time_ns include their host launch wrappers. Working buffers total401,808B.

| Version | Mean2–5 (µs) |
|---|---:|
| CUDA |190.062|
| CUDA.jl |287.548|
| KA |330.403|
| JACC |662.341|

Native uses constant symbols; Julia uses global coefficient arrays read-only by
code. Source logical16×16 tiles are preserved. KA uses static local memory1296B;
native/CUDA.jl/JACC use dynamic1296B for three stages. JACC.shared adds a template
preload and barrier before source-image overwrite. The source threshold kernel
has no shared-memory reads; its original unused1296B reservation is retained in
archived H1296 runs. The fresh symmetric primary prospectively selects H0 before
target translation. This is integration/resource adaptation, excluded from
untouched-native aggregates, not a mathematical kernel correction or mandatory
framework policy. Native function attributes (registers18/24/18/8, static/local0)
are compiler/runtime queries, not profiler counters. No roofline, occupancy,
cache-regime or causal frontend-performance claim follows from these timings.

## Cold reproduction

Pinned Julia1.12.6, CUDA.jl6.2.1, KA0.9.45, JACC1.4.0, local CUDA12.0,
nvcc12.0.140, V100sm70. The HPC SDK12.9 compute-sanitizer was a diagnostic tool only;
it does not change the selected build/runtime. Source environments and all model
source dependencies/assets are hash-bound in the receipt.

CPU input/oracle binaries are deterministic and locally retained; their generator,
exact hashes/byte counts and Python3.12.3 provenance are committed instead of
requiring raw GPU captures or large regenerated binaries in source Git.

```bash
python3 src/ced-julia/generate_assets.py
python3 src/ced-julia/prepare_cold_config.py /tmp/ced-cold.json
python3 tools/verify_portable_julia.py ced --config /tmp/ced-cold.json
```

The verifier owns the GPU6 lock; do not wrap it in another flock. The cold config
removes all reuse and reruns four fresh series, while archived primary samples
remain immutable. Installed JACC API dependency paths must exist with the pinned
hash. `audit_saved_outputs.py RESULT` performs the CPU-only full saved-artifact
and17-file byte-exact cold-regeneration audit; no GPU work is added.

## Effort and research records

Actual target usage windows and calibration adjudications live under STATE
`usage_boundaries_300/ced` and `calibration_537/ced`; manifests under
`.porting-state/porting_logs/ced-julia-{cuda_jl,ka,jacc}` retain candidate/check costs.
CUDA.jl creates shared math/host helpers; KA then JACC reuse them. Counts are
context/cache/session inclusive, not exclusive billing or historical recovery.
Shared readiness, verification and annotation are separate phases. Failed setup,
CPU-audit syntax and unintended duplicate native-series events are retained
separately from target compiler feedback. `paper_observations.*` links RQ1–RQ5.
