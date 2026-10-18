# D3Q19 BGK: a defined moving-wall cavity core

The named primary study advances the source17³ cavity for20 steps from rest, using19 populations per site, local bounce-back, moving zmin wall and source Re100 relaxation. Both population buffers and the actual velocity field are retained. The historical Julia entrypoint was an unrelated mcpr alias; its bytes, Makefile and target snapshots remain archived.

## Conditional reference and scope

Native setup allocates boundary direction bits without initialization before find_wall uses bitwise OR; unused direction fields are also undefined. The study explicitly seeds both to zero before calling the unchanged original setup kernels. This initialization-only repair is declared before the first measurement, hash-bound in reference_provenance.json, and excluded from untouched-native aggregates. No undefined GPU run is invented. Original native source and Makefile are also archived intact. Julia models use identical frozen CPU-generated metadata, verified against actual native GPU setup. The setup kernels themselves are not translated or timed.

All four models execute the same20 collision/stream steps, population pointer swaps and completion synchronization; out_u=true at every step to retain velocity. This differs explicitly from the source CLI's sparse output schedule. Initialization, resets, compilation warmup, validation and serialization are outside timing. Five fresh trajectories are checked; only trial1 timing is omitted from the mean. This is not the source2000-step CLI, N102 regression, D3Q27, steady-state or continuum/grid-converged validation.

## Precision and numerical evidence

Float64 populations coexist with Float32 macroscopic velocity and equilibrium subexpressions. The independent CPU oracle uses generic velocity-vector pull streaming/reflection, math.fsum moments, Float32 velocity rounding and the standard second-order vector equilibrium, rather than importing GPU helper formulas. Predeclared atol1e-7+rtol1e-6*abs(ref) applies to evolving fields; all112 physical guards, immutable rho and complete boundary metadata are exact. Every trial retains329287 Float64 values, including an864720-byte exact actual initial GPU-state prefix.

Full current/previous populations and actual u_star agree with both references and the independent discrete oracle. Positive populations and relative active-fluid mass drift below1e-6 are checked separately. CPU history and an analytic first-step moving-wall momentum sign witness are preserved. Raw relative-only errors can be enormous near exact-zero reference values; output_audit.json reports full-field absolute error, all-value hybrid-tolerance utilization, and clearly separated nearzero diagnostics without dropping values from acceptance.

CUDA.jl/KA/JACC share one source-derived ordinary Julia/StaticArrays cell helper with genuine framework dispatch. No CUDA-specific device intrinsic is needed. All use source-matched3D64-thread x blocks, with17 active x positions. Native unrolled code/launch-bounds hints differ from Julia tuple-loop forms; register/spill/occupancy behavior is unprofiled. Host-inclusive timings do not isolate bandwidth, device time or intrinsic framework causality.

The first reference candidate failed Julia frontend syntax before device compilation; candidate2 passed after explicit floating literal fixes and source precision review. KA/JACC first candidates passed. Phase-specific attempts and usage are separate; historical translation cost is unknown and shared helper work is counted once.

A periodic/nonuniform operator control is planned as a separate optional secondary after primary queue coverage. It has not run and is not included in primary performance. Recheck saved fields with audit_saved_outputs.py; use the central serialized GPU6 runner for any future GPU work. Root owns Git commits.
