# SI: set-intersection counts through cuBLAS

The source computes pairwise integer-set intersection cardinalities using CPU binary-incidence packing and cuBLAS SGEMM. Source and all ports have **zero authored GPU kernels**. CUDA.jl, KA and JACC provide genuine allocations, with explicit CUDA host/runtime and vendor integration. No dummy kernel or independent GEMM implementation is claimed.

## Verified scope

| Case | Sets / effective universe / partition | Qualification |
|---|---|---|
|Original uniform dataset|1000 /1001 /1000|Primary, original dimensions unchanged|
|Heterogeneous full tiles|24 /17 /8|Verification control, excluded aggregate|
|Heterogeneous partial tiles|19 /17 /8|Corrected-reference secondary|

The actual original generator resets its seed per set: all1000 sets have the same95 members, with identical ascending/descending files. The original CLI passes all499,500 distinct pairs. Heterogeneous controls include empty, universal, disjoint, identical and variable-size sets.

The unchanged partial CLI fails42/171 pairs (max error10). B is packed with strideP but SGEMM uses B.length; the output writer expects strideP while SGEMM uses B.length. The separate owned correction sets lda/ldc=P and zeroes unused padding for complete capture. Individual contributions were not isolated. First full tiles initialize output, and later unused partial slots can retain prior results; **no observed uninitialized-memory/sanitizer failure** is claimed. Original CUDA files are unchanged.

All five trials of all four models and all three cases are checked. All 60 complete captures,240,462,160bytes, are byte identical across models and match independent Python set intersections. Every actual A/B device input, all P2 output slots/diagonals and17tailguards are captured. Counts<=1001 are exactly representable in Float32. This certifies the named set-count operators, not application quality or every CLI argument.

## Measurement

Mean trials2–5 milliseconds: native4.68886975, CUDA.jl5.351017, KA4.954629, JACC5.37619925.

Completed ROI segments include allocation, handlecreate, CPU packing, H2D, SGEMM, result D2H, clears, frees and handledestroy. Actual input/guard readbacks are excluded. Whole-device fences precede each pause and follow excluded guard initialization. Clock/readback cache effects, host API/GC and allocator differences remain covariates; this is not uninterrupted original-CLI or isolated-GEMM timing.

The initial CUDA.jl control was numerically exact but had unproven H2D completion before a pause: its API submits asynchronously. All v1 source/results/times remain preserved with hash-bound aggregate exclusions. Fresh symmetric v2 series use explicit completion fences. This is a measurement repair, not a mathematical/compiler repair or fastest selection. Nativev1 blocking-copy times were not individually invalidated, but are superseded for the matchedv2 cohort.

The same bound cuBLAS12.0.2.224 uses mathmode0, host pointermode0 and legacy queue. Julia poolnone, frontend undef constructors and explicit CUDA.unsafe_free! preserve real allocation/release work. The class is vendor_library_interop with timed CPU processing separately recorded. Internal vendor grids, precision and counters are opaque/unprofiled.

## Evidence and effort

Final receipts are under .porting-state/portable_julia/runs/:

- 20261020T035501829699Z_si/result.json: primary
- 20261020T035457963905Z_si/result.json: fulltile control
- 20261020T035505977492Z_si/result.json: corrected partial

saved_audit_final_*_v2.json, scientific_witness.json and final_source_binding_audit.json bind all numerical evidence and47 source/dependency files. study_adjudications.json preserves scope/exclusions without modifying immutable executions. Original failure and corrected CLI proofs/patches remain under native_diagnostics/ and native_corrected/. Raw CLI/GPU outputs stay local; hashes do not reconstruct them.

observed_token_windows.json distinguishes native preparation, initial CUDA translation, shared timing repair, remaining CUDA verification, KA/JACC, final verification and annotation. The initial CUDA window is partial and its END shared final metadata/archive writes, potentially omitting that generation. Native END likewise shared final readiness metadata. Raw counters remain unchanged. Other ENDs are separate rounds. Cached/reasoning counts are subsets, not additive exclusive billing or historical recovery. KA calibration applies only to ordered vendor orchestration; CUDA/JACC are conservatively excluded. Program checks differ from device-kernel/JIT counts.

## Cold reproduction

All8 CPU datasets/oracles/manifests regenerate exactly. No irreplaceable binary is needed. A standalone C++ recipe was byte-matched against the actual original generator, preserving seeded libstdc++ draws and bit-index-to-ID convention. Checked environment: GCC13.3/libstdc++ and NumPy2.5.4; other STL implementations can fail explicit SHA checks.

From repository root with pinned Julia projects, CUDA12.0 and exact system library:

    PYTHONPATH=/tmp/pasc-julia-plot-tools python3 src/si-julia/regenerate_assets.py src/si-julia
    cmake -S src/si-cuda -B /tmp/si-cold/build -DSM_ARCH=70 -DCMAKE_CUDA_COMPILER=/usr/bin/nvcc -DCMAKE_BUILD_TYPE=Release
    python3 src/si-julia/prepare_cold_configs.py --source-root "$PWD" --native-deps /tmp/si-cold/build/_deps --output /tmp/si-cold/configs
    PYTHONPATH=/tmp/pasc-julia-plot-tools python3 tools/verify_portable_julia.py si --config /tmp/si-cold/configs/si_cold_original_uniform1000_v2.json
    PYTHONPATH=/tmp/pasc-julia-plot-tools python3 tools/verify_portable_julia.py si --config /tmp/si-cold/configs/si_cold_heterogeneous_full24_v2.json
    PYTHONPATH=/tmp/pasc-julia-plot-tools python3 tools/verify_portable_julia.py si --config /tmp/si-cold/configs/si_cold_heterogeneous_partial19_v2.json

CMake fetches pinned user-space sources; no OS installation/privileged command is needed. The runner owns the GPU6 lock; never add outerflock.

The measured harness fixes /usr/lib/x86_64-linux-gnu/libcublas.so.12.0.2.224; the helper verifies its exact SHA. This system-library path is a reproduction prerequisite, not universally relocatable. Julia API paths resolve by saved SHA/pinned projects. Fresh cold configurations contain no reused results. CPU cold regeneration and current-root config preparation passed; **fresh relocated-host GPU reproduction was not executed**. Raw GPU data and compiled binaries are not source-Git assets.

Original GPLv3 license is preserved as LICENSE.source.
