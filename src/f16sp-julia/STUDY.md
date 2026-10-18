# f16sp: precision policies, genuine vendor calls, and mathematical accuracy

## Covered operator paths

All 17 native mode/grid paths are implemented: three distinct custom dot kernels at five grids, actual cuBLASDotEx_64, and actual CUBDeviceReduce. The named scope is 32,768 half2 pairs (65,536 Float16 scalars/vector), 128 threads/block, grids 256/128/64/32/16, two input fixtures, and R3. The original 134,217,728 half2 size, million-block grids, and R1000 are unmeasured. One explicit warmup per source path replaces the original 1000 untimed warmups.

The custom policies are:

1. Float16 fused multiply-add and paired Float16 shared-memory tree, then Float32 pair combination/global atomic addition.
2. Float32 FMA and paired Float32 shared tree.
3. Float32 FMA, lane-pair combination, warp shuffle offsets 1/2/4/8/16, then sequential four-warp aggregation, matching the source CUB hierarchy.

cuBLAS computes in Float32 and returns Float16. CUB rounds each half product, adds lane pairs in Float32, and reduces in Float32. These approximate operators are distinct. The historical Julia implementation reused one FP32 kernel for all three and printed cuBLAS/CUB labels for framework sums. It is preserved; the current reference restores the real algorithms and APIs.

## Numerical agreement versus mathematical accuracy

An independent Float64 evaluator explicitly rounds after every modeled Float16/Float32 FMA and reduction step. The fixture exponent span keeps the oracle operation exact before each rounding. Source-policy agreement uses predeclared 0.02 + 2e-6 × |policy oracle|, admitting Float32 atomic-order variability; the budget is unchanged after results. cuBLAS's half result is additionally exact against the half-rounded ideal.

Each of 20 artifacts has 262,168 verified input-bit/configuration prefix values, 34 complete scalar outputs, and 7,616 exact guards. Native reads inputs back and proves complete byte equality before storing equal fixture bits; Julia directly serializes readback bits. Both recheck inputs after all modes. This establishes actual input identity without claiming that native literally serializes its readback buffer. The native source exposes scalar answers, so there is no invented block-partial-array evidence. Guards and prefix are exact independently of payload tolerance. Relative errors are computed with an explicit epsilon floor, not uncomputed zeros.

The saved audit checks all artifacts, component times, and 49 current source/fixture/library bindings. Mathematical errors are separately retained in all 680 rows of precision_quality.json:

- The signed fixture's ideal dot over actual quantized inputs is -0.1533203125. Faithful FP16-policy outputs vary with grid from approximately +0.5298 to -3.8262. Half-rounded-product CUB gives approximately +0.5323. These are substantial arithmetic-policy accuracy costs, not translation failures.
- The uniform generator's actual quantized-input dot is 65472.015625. The source always prints error against its prequantization analytical target 65504. Some FP16 paths happen to return 65504 and appear best by that printed baseline while being less accurate for the actual input than cuBLAS's 65472.

The study validates source-policy reproduction and reports mathematical accuracy separately. It does not certify a scientific application's or trained model's quality.

## Per-component framework boundaries

The complete program is mixed_kernel_library_pipeline.

| Component | CUDA.jl | KA | JACC |
|---|---|---|---|
| Half and FP32 paired trees | framework_device_kernel | framework_device_kernel | hybrid_device_intrinsics |
| CUB-style warp hierarchy | framework_device_kernel | hybrid_device_intrinsics | hybrid_device_intrinsics |
| Actual cuBLAS/CUB APIs | vendor_library_interop | vendor_library_interop | vendor_library_interop |

CUDA.jl uses its own shared memory, atomics, and shuffles. KA paired trees use localmem/synchronize/Atomix; its warp path calls CUDA shuffles explicitly. JACC callbacks use CUDA static shared memory, barriers, atomics and shuffles. This is a disclosed mapping choice, not proof that no alternative portable algorithm exists or an unsupported outcome.

The vendor bridge invokes the exact original Half2Mul/CUB operator and real cuBLAS API through a CUDA-specific C ABI. Vendor paths are not claimed as independent portable-library rewrites. Native packed half2 instructions versus scalar Julia half operations, CUB templates versus explicit warp code, shared layout, registers, launch/reset wrappers and vendor calls are performance covariates. Julia shared allocations are 512/1024/16 bytes for the three custom modes; native CUB storage/register use is unprofiled. Dynamic shared memory is explicitly zero. Vendor internals are opaque beyond reviewed source/API evidence.

## Observed toolchain and loader resolution

CUDA12.0.140 rejects the unchanged original host/device Half2Mul because __hmul2 is device-only in those headers. Exact error, source and build records are archived. CUDA12.9.41 sm70 builds the same unchanged source: no qualifier, arithmetic or native kernel patch is needed. This is an explicit separate toolchain stratum, not a V100 hardware or KA/JACC feature barrier.

The bridge is CUDA12.9-built with default static-runtime linkage, and its complete binary hash is retained. Julia custom-kernel package/runtime preferences remain CUDA.jl6.2.1/local12.0. Native reports runtime12090. The bridge's static-runtime version was not separately queried.

An actual loader audit found ELF interposition: the first numerically passing Julia reference bound system cuBLAS12.0 despite bridge RUNPATH12.9, while native used12.9. That series remains immutable and was excluded before controlled v2, not because of speed. LD_PRELOAD before Julia aligns all four implementations to the actual cuBLAS12.9.0.13 path/version/SHA. Every model reports the same resolved library hash. Unchanged native first-series samples are reused with explicit cross-configuration provenance. CUB has the same original operator/API and CUDA12.9 headers; embedded native versus shared bridge binary/code generation is not asserted identical.

## Compiler loop and effort

One genuine CUDA.jl device-program candidate has two definitions and three type-specialized semantic modes. Three process evaluations occurred: shared host-parser failure before authored JIT; numerical pass with vendor interposition; controlled aligned pass. Thus one host syntax repair and one environment-alignment repair are separate from the proactive historical semantic restoration. Two processes reached authored JIT checks. Shared bridge compilation checks were CUDA12.0 fail and CUDA12.9 pass; the native adapter built once.

KA and JACC each passed their first authored program and actual compile/execution evaluation, with no feedback repair. Each has two definitions and three specializations. Five grids, 34 evaluated cases, repetitions and outer trials are not translation attempts. Internal compiler/cache jobs remain unknown. A postverification annotation orchestration expression was rejected before any shell execution; it is separately recorded as tooling effort, with no device attempt or measured-source change.

Historical attempts and tokens remain unknown. Shared restoration, oracle, bridge and loader work are separate from target phases. Usage counters include context/cache/coordination and overlap the broad workflow; they are not exclusive billed translation tokens. Saved-source proxies include mechanically reused bodies.

## Timing and reproduction

Each component times reset+custom launch or the actual vendor API. CUB includes temporary query/allocation/free as in source. Whole-device completion covers vendor legacy stream0 and Julia task streams. Inputs, readbacks/checks and guards are outside ROI. Five checked in-process trials are retained; the report averages trials2–5. Total is the sum of 34 per-repeat component means, not one uniform-work kernel. Separate paths remain in raw records and component_timings.json.

| Native CUDA (µs) | CUDA.jl (µs) | KA (µs) | JACC (µs) |
|---:|---:|---:|---:|
| 317.3433333 | 352.5920833 | 431.6336667 | 606.9563333 |

No pure-abstraction causal performance claim, bandwidth attribution or roofline position follows from these small cache-resident fixtures without counters.

Final selected bundle: .porting-state/portable_julia/runs/20261018T070105511666Z_f16sp/result.json.
Contract: study_config.json. Four essential CPU fixtures/oracles are study_case{1,2}_u16.bin and oracle_case{1,2}_f64.bin, with generator/hash provenance. The vendor shared binary is locally built from tracked source, not a committed GPU-output archive. make run builds it if absent and performs a fresh four-model verification without reusing selected timing series. Coordinator owns separate verified-variant commits and paper integration.
