# RMS normalization scope

Original native Float32 two-scan kernels and their XOR warp reduction, shared 32-word reduction and vector loads/stores remain unchanged. All six source block sizes32,64,128,256,512,1024 run for H1024/vector4 and H1027/scalar, with seven/five rows and signed varied/unit gamma fixtures. Zero, constant, sparse spike and epsilon-dominated rows supplement varied signed rows. This does not certify uninstantiated FP16/BF16 templates.

Every output is compared with an independent Float64 fsum/sqrt calculation using the actual Float32 epsilon. Predeclared finite-case budget1e-5+2e-6|oracle| and dual budget2e-5+4e-6|reference| retain Float32 rsqrt/FMA/tree rounding. Input/gamma/initial output and all guards are checked exactly. Old Julia marker and fabricated timing output are preserved, not accepted as numerical evidence.

Julia uses native CUDA XOR/rsqrt intrinsics inside genuine CUDA.jl/KA/JACC dispatch and retains grouped4/scalar mapping. Mark these as hybrid_device_intrinsics; no cross-backend portability claim. Shared32 uses framework synchronization. Reinterpreted NTuple4 VecElement views preserve16-byte vector layout; generated code proof is separate. JACC shared-template preload is an additional backend operation and performance covariate.

Timed region sums24 separate kernel-launch/context-completion windows, each divided by R20 overwriting launches. Five trials are fully checked, discard only first timing, average2–5. Setup/reset/copy/JIT/capture outside. Per-path times retained; no observed traffic/bandwidth/roofline claim follows from source access estimates.


## Model-relative classification correction

Model-relative descriptor correction: CUDA.jl kernels using their own public device atomics/shuffles/pointers/shared APIs are framework_device_kernel; KA/JACC calling CUDA-specific APIs remain hybrid_device_intrinsics. No foreign device ABI in these CUDA.jl bodies; preserve CUDA-specific feature tags. No executable source, acceptance, artifacts, trials or timing change; zero GPU reruns/translation attempts.
