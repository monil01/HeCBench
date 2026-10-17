# Verified named Float64 strided-batched matrix-vector study

Use study_config_ka.json for KA-first evidence and study_config.json for the JACC extension with unchanged reference/KA series reused. Configuration is square32, n1, leading dimension64, batch128, Float64 only. Native and CUDA.jl call actual cuBLAS; KA/JACC use custom row-dot kernels. Full CPU/dual-reference artifacts include active outputs and padding. Float16/Float32 and other source CLI cases remain unmeasured.

Native source includes missing reference.h. Study build supplies -I to the tracked sibling blas-gemmBatched-cuda/reference.h, solely to resolve the original CPU helper declaration/definition; native source is unchanged. The failed native compile, exact include path and helper hash are recorded in local reference audit/events. No unsupported-framework claim is involved.

Library selection, vendor opacity, custom mapping and cache-resident size limit are explicit performance confounds. All clocks use five warmed in-process trials, averaging2–5, with ten internal operation repeats.
