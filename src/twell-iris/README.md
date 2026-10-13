# TwELL IRIS dmem HIP port

Preserves the native HIP matrix-core D2T packing kernel, both source gated T2D BF16/fp32 kernels (22/32 feature tiles), GPU reference kernels, deterministic GPU hash fill, and full validation/stack benchmark flow. No new mathematical solver replaces the native implementation. IRIS DMem owns all device buffers and tasks dispatch all kernels; host-side weight caches are represented by DMem objects.

CUDA host arguments/defaults, seeds, shapes(hidden2048/features5632 or8192/M multiple256), negative gate bias, packing budget31, validation error threshold3%, layer/warmup/repeat/ping-pong behavior and run target --reps100 remain. Original invalid nonpositive workloads are rejected. HIP kernels require wave64 (probed on GPU). Existing native HIP adaptation replaces Hopper WGMMA/TMA with rocWMMA MFMA; that native algorithm is retained here, not independently reimplemented.

Build make; run IRIS_ARCHS=HIP ./main [original options]. Configure IRIS_ROOT/IRIS_BUILD/CXX/HIPCC. Installed rocWMMA headers are needed for backend compilation. Optional HECBENCH_DUMP_PREFIX exports complete validation reference/output/packed words and final stack output for observational comparisons. Unused packed payload slots are initialized on IRIS; only counts/used slots define source semantics.

GPU validation evidence is recorded locally. Full production defaults are not claimed tested. CUDA backend and non-wave64 HIP architectures are not validated. Timing includes synchronous task overhead; no native-throughput equivalence claimed.
