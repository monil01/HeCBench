# Prospective electrostatic grid case

All four models pass the named full-array Float32 Coulomb-potential contract against actual input readbacks, exact guards, analytic single-charge, exact paired cancellation, zero-charge, and independent Float64 direct-sum checks. Three Julia outputs are bitidentical; native agreement is tolerant. The 4001-atom fixture exercises original4000+1 chunking.

Five trials are checked for every model; mean timing excludes only trial1. Native/CUDA.jl/KA/JACC means are0.773907/4.180167/4.686892/4.723239ms for six kernel+completion fragments. This one-CTA64x8 grid is not the original default768x768/one-million-atom pipeline. Native constant-memory access and rsqrtf differ from Julia array access and inv(sqrt); these are explicit covariates, not established causal abstraction costs. No measured roofline claim.

CUDA.jl/JACC passed their first complete candidate. KA candidate1 was rejected by the frontend macro for explicit return; removing return produced candidate2, with one actual GPU-JIT process and no device algorithm repair. Retained failed source and diagnostic are linked in the KA receipt.

All three phases have actual input/cache/output/reasoning counters. Root coordination and shared tooling interleave these windows, so they are retained as scoped observations and excluded from clean per-target calibration. Historical missing token counts remain null. See paper_observations.json for exact evidence, metrics and RQ mappings.
