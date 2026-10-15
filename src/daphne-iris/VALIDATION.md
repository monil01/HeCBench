AMD HIP validation
==================

The semantic reference is the current `../daphne-cuda/main.cu`: 100000
LCG-generated points, per-batch seed, extrinsics, distortion, projection and
visibility filtering. Its default one batch and `-p` semantics are unchanged.
The existing daphne-hip sibling is a different, legacy data-driven points2image
program missing common sources/headers/data. That legacy workload is NOT
validated or repaired here, and no legacy native execution is claimed.

`make native-reference` compiles the **unmodified current CUDA translation
unit** with hipcc. The validation-only native_hip_compat/cuda_runtime.h maps
allocation, copy, synchronization and free calls to checked real HIP calls.
Its original kernel actually executes on the MI100. There is no CPU kernel
fallback or substituted GPU result. Optional DAPHNE_DUMP captures input/output
at actual transfers, without changing source math, launch or input data.

The separate IRIS main owns all device data through DMem and launches the
original projection using an actual IRIS HIP GPU. It checks task/update/flush
status and destroys DMem before releasing host storage and the runtime.
Set DAPHNE_DUMP to an existing isolated directory for whole-output artifacts.

Actual tests: default, -p1, -p3, -p7 (twelve batches, 1.2 million points).
All 14.4 million input/output entries are finite and native/IRIS byte-exact;
all validity flags are exact. Both programs' original host check prints PASS.
verify_projection.py independently evaluates every projected coordinate in
double precision, maximum absolute error3.03e-4 below the unchanged1e-3 bound.
Original generated data exercises the visible branch; broader input coverage
is not claimed. No absent dataset is synthesized for the legacy workload.

Native timings are kernel timings plus original harness overhead; IRIS timing
includes task submission and flushes. Wall times include validation artifact
I/O and are diagnostic, not an unqualified production performance comparison.
Builds/runs/failed histories/manifest/checkpoint are retained locally under
`.porting-state/porting_logs/daphne-iris/attempts/attempt_04`.
