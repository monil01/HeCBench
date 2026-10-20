# Optical flow — complete source solver and full original image pair

## Scientific scope

The named primary study runs the tracked **640×480** `frame10.ppm` / `frame11.ppm` pair, all **five pyramid levels**, **three warps/level**, **500 Jacobi iterations/warp**, source `alpha=Float32(0.2)`, and all six GPU stages. The complete `ComputeFlowCUDA` function boundary includes allocation, copies, normal diagnostics, kernels and frees. This is 7,576 primary GPU launches; it is not a small solver surrogate. SDK file searching/loading and FLO-file serialization outside the original GPU timing boundary are not included in performance claims.

Additional checks retain a 65×49 partial-tile identity case with three levels and the same warps/iterations, all six primitive operators, and a nontrivial 65×13 Neumann Fourier-mode manufactured Jacobi system. Every trial executes 12,626 checked GPU launches across these cases; controls are outside the reported primary time.

`original_native/` preserves all 22 tracked files, including image assets and source notices. The tracked native baseline already uses `tex2d_compat.h` software normalized bilinear/mirror interpolation. This task did not replace hardware textures. The native six GPU headers remain byte-identical; `native_source_preservation_audit.json` reverses the host observation rewrites exactly. No native scientific arithmetic repair was required.

Input row padding is explicitly initialized to zero in shared fixtures. Original loaders leave it undefined; it is not a scientific image pixel. Complete source stride operations remain unchanged. Dead scratch/return padding is retained raw with its definedness qualification, excluded from numerical identity claims. The original `WriteFloFile` explicitly writes only `x < width, y < height`; the complete active-field contract follows those scientific file outputs. Pre-existing `src/opticalFlow-cuda/FlowCPU.flo` / `FlowGPU.flo` and generated binaries are untouched and are not current verification evidence.

## Accuracy and physical evidence

Final receipt: `.porting-state/portable_julia/runs/20261020T023518762905Z_opticalFlow/result.json`; config: `study_config_jacc.json`.

All five trials/model retain 1,467,732 Float32 primary values: an **exact 818,494-value actual input prefix**, all active final `u/v` fields, and all eleven primitive/control output arrays. All three Julia scientific artifacts are bit-identical. Maximum native-versus-Julia absolute difference is **5.82e-5 pixels**, inside the prospectively fixed `atol=.002, rtol=.0002` contract. This is complete field agreement, not membership/score/property-only verification.

An independent scalar **Float64 implementation of the complete GPU equations**, without source solver/CUDA/Julia inclusion, provides the full-field reference. The largest native error is **0.000444883 pixels**; Julia versions reach **0.000456804 pixels**, under the predeclared `.05 + .001*abs(reference)` arithmetic-agreement budget. This budget is not a universal rounding-error theorem or physical motion endpoint guarantee.

Independent source-equation primitive bounds, full one-step and 500-iteration manufactured Jacobi arrays, and the identity case also pass. The independently generated manufactured Fourier solution differs from its closed form by at most 1.60e-8 before GPU execution. Each model has **3,246,190 independent scientific-value comparisons** over five trials.

Each trial additionally retains:

- complete physical payloads and **28,160 actual guard bytes** for all **55** GPU allocations, with allocation IDs, sizes, offsets and hashes;
- complete padded host-return buffers, including dead bytes;
- exact joins between final GPU physical buffers and full host returns, and between actual finest-level GPU images and shared fixtures.

The active scientific fields are fully retained. Dead padding is not silently replaced with zeros or claimed identical across processes: `raw_output_identity_claim=false` records that qualification.

### Why a second CPU reference matters

The original CPU reference passes its source mean-L1 threshold: native full-scene L1 is **0.0438767 < 0.05**. Its interpolation boundary policy differs from the tracked GPU software mirror function. An exact-source CPU witness shows `[0,1,2]` sampled at `x=-0.25` gives `0.25` in the old CPU function versus `0` with periodic mirror interpolation; at `3.25`, it gives `2` versus `1.75`. The witness, original source and log are preserved.

This local difference does not establish wrong native GPU flow or the cause of the entire CPU/GPU discrepancy. The stronger Float64 reference follows the GPU's actual mathematical boundary policy. Neither reference establishes true scene motion, regularization adequacy, convergence, or application accuracy without external ground truth.

## Implementations and mapping qualifications

Each Julia model owns six entries and dispatch plus a complete Julia host pipeline. Later models reuse validated mathematical bodies and host logic; they are ordered translations, not independent generations.

- **CUDA.jl:** two 272-element static shared arrays and the exported workgroup barrier preserve the source tile/halo organization.
- **KA:** two `@localmem` arrays and `@synchronize` use the same padded 32×6 groups and source halo ownership. All lanes load and synchronize before active-output guards.
- **JACC:** a lazy isbits `AbstractMatrix` represents two stacked 34×8 halo planes. Public `JACC.shared` stages the bounded 34×16 matrix with all 192 threads and one barrier; two linear views feed the unchanged output equations. It uses **2,176 dynamic shared bytes**, explicitly requested. Other kernels request zero; launch specs are fresh. No CUDA device extension or full-image shared-memory copy is hidden here.

Native source has `volatile` shared accesses; Julia uses regular typed shared arrays plus barriers. JACC also changes cooperative load ownership and nominally reads eight unused corners (544 indexed reads versus 536 source assignments). These are explicit mapping adaptations, **not** instruction equivalence or unsupported-framework claims. Nominal source reads are not measured DRAM transactions.

## Timing and feedback

Full warmup, five checked trials, average only times 2–5:

| Version | Mean seconds, full function | Complete-program / actual GPU-JIT processes | Actual translation-window tokens |
|---|---:|---:|---:|
| CUDA | 0.048665726 | 1 GPU process / 1 build invocation | Readiness separately: 3,453,832 |
| CUDA.jl | 0.043134132 | 2 / 1 | 3,555,914 |
| KA | 0.060736139 | 1 / 1 | 3,433,290 |
| JACC | 0.210054291 | 1 / 1 | 2,978,885 |

CUDA.jl's first host bundle had two syntax errors and was rejected before GPU compilation. Its corrected host bundle passed with the same six GPU mathematical bodies. Standalone parser checks, whole-program evaluations, actual JIT processes and backend compiler invocations are distinct units; the last remain unknown. A shell orchestration error submitted the already parser-rejected first bundle and is retained separately. Native CPU-reference preparation also needed a forced CUDA-header include when using g++ rather than nvcc; that is a shared checker build issue, not a native GPU failure. Annotation-helper failures are separate from target compiler attempts.

Timing includes source allocation/copy/free work, but guarded allocations, Julia allocator/pool choices and explicit completion before paused raw capture are covariates. These timings are excluded from untouched whole-CLI aggregates. Repeated JACC 2D launch-policy/attribute checks, dynamic shared staging and loader differences are source-observed candidate mechanisms for its higher measured time. No isolated control, profiler counters or roofline analysis establishes causality. CUDA.jl's lower time likewise is not claimed to arise from removing `volatile` or from a generally superior compiler.

Tokens are prospective context-inclusive workflow counters, including repeated/cached context, API review, feedback, verification and coordination while sharing the GPU. Native preparation and annotation are separate; nested verification windows overlap translation and must not be added. CUDA.jl creates the common harness, later targets inherit it, and JACC's capability review considers but does not implement an earlier vector-loader idea. Counts are not billing, exclusive coding cost or recovered historical effort.

## Replay

Existing prerequisites: nvcc/CUDA headers, g++, pinned Julia environments (Julia 1.12.6, CUDA.jl 6.2.1, KA 0.9.45, JACC 1.4.0) and the eight fixture/reference files. Essential assets and hashes, including archived original PPMs and the small boundary witness log, are listed in `essential_assets_for_commit.json` (about 13 MB). No privileged command, operating-system mutation or package installation is needed.

Recorded configurations contain absolute paths and source-bound reuse receipts:

```sh
python3 tools/verify_portable_julia.py --config src/opticalFlow-julia/study_config_native.json
python3 tools/verify_portable_julia.py --config src/opticalFlow-julia/study_config_cuda_jl.json
python3 tools/verify_portable_julia.py --config src/opticalFlow-julia/study_config_ka.json
python3 tools/verify_portable_julia.py --config src/opticalFlow-julia/study_config_jacc.json
```

For fresh/relocated runs, update source/executable/evidence paths and remove `reuse_result`/`reuse_models`; preserve inputs, tolerance and timing scope. On this shared host use only the central serialized GPU6 runner.

CPU-only asset generation and full evidence auditing:

```sh
python3 src/opticalFlow-julia/prepare_runtime_assets.py --output-dir /tmp/optical-assets --cxx g++ --cuda-include /usr/include
PYTHONPATH=/tmp/stencil_ports_numpy_226 python3 src/opticalFlow-julia/audit_complete_evidence.py --result .porting-state/portable_julia/runs/20261020T023518762905Z_opticalFlow/result.json --model jacc --folder .porting-state/portable_julia/opticalFlow_primary --output /tmp/optical-audit.json
```

The existing NumPy path is illustrative; use an evaluator's installed environment (recorded NumPy 2.2.6). Fixture-only regeneration was rerun and matched both measured inputs byte-for-byte; individual CPU reference builders were executed, while the new combined generator's full orchestration was not rerun after GPU verification. Missing/short files fail closed. Keep raw captures, binaries, unrelated original FLO outputs and pycache outside Git/Overleaf.
