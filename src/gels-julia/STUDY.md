# gels: vendor batched square-system interoperability

The native source and historical Julia both invoke cuBLAS batched GELS for two5x5 systems in Float32, ComplexF32, Float64 and ComplexF64. The measured fixture is square and full rank: it does not establish overdetermined least-squares behavior. Complex inputs are the original purely imaginary A and B, with real unit solution. All three Julia versions use genuine model allocation, then the same pinned system cuBLAS12.0.2.224 raw ABI. There are zero authored solver device kernels; vendor internal kernels, resources and accumulation policy remain uninspected. These timings belong to the library interoperability stratum, separate from framework kernel comparisons.

## Scientific numerical witness

Every solution's real and imaginary components, host status and host info are retained for all four types in all five trials. The504 input components/controls match exactly. An independent Fraction Gaussian elimination solves the actual rounded input values; complete complex-modulus error obeys the original per-type1e-6 FP32/1e-8 FP64 bounds. This strengthens the original projected `sqrt(abs(real(r*r)))` check, which could cancel complex error components. Maximum error against the exact actual-input solve is6.73532484e-07; maximum normwise infinity backward error is7.69893204e-08. Conditioning and expected-one rounding discrepancy are reported separately in the witness. OriginalNULLper-batch info buffer is preserved; individual batch info is not separately captured.

## Scope and timing repair

Pointer arrays, allocation, input reset/transfers and handle setup are outside timing. Each trial sums four typed vendor-call regions, with explicit context-wide completion before/after each call. An initial Julia/KA series used no-argument CUDA synchronization, which waits the task-local stream while raw cuBLAS handles use default stream0. Those captures matched numerically but are timing-invalid. Their original logs/artifacts are preserved under S2. The corrected series reruns affected Julia/KA models and reuses only unchanged native evidence; JACC begins after corrected KA verification. This is a necessary measurement correction, not a favorable timing selection.

The named raw ABI also removes the historical Julia high-level wrapper's pointer-array/info-buffer setup from its timed region, matching native setup boundaries. All Julia timed vendor-call code is identical; framework allocation differences occur outside the timer. Consequently these values are not evidence that a framework implements a faster solver. Five trials retain all outputs and average trials2–5. Mean microseconds: {"cuda": 144.689, "cuda_jl": 189.6765, "ka": 190.16375, "jacc": 182.6165}. Result: `/home/imo/HeCBench_julia/.porting-state/portable_julia/runs/20261017T224329871669Z_gels/result.json`.

Shared effort includes two failed native wrapper compile diagnostics for a template/ternary parse before any GPU execution, followed by the raw-library synchronization repair. Prospective activity windows include repeated/cached context, evidence/ownership coordination and related audit discussion; they are not exclusive coding or billed cost. The parent workflow contains its children; do not sum them. Main-only source proxies omit the shared ABI/harness.


Prospective rawusage intervals include cached/repeatedcontext and coordination; the parentworkflow contains its childphases and must not be summed with them. Exclusive billed translation cost remains unknown; source-size proxies remain separately labeled.
