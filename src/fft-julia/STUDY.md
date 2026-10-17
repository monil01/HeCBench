# FFT study

## Measured operation

The original Makefile defaults to double precision. The measured configuration uses 128 independent 512-point ComplexF64 FFTs (1 MiB), equivalent in size to source selection0. All four implementations execute the same three radix8 stages,64 threads per FFT,8 complex values per lane, padded576-double shared scratch, and normalized inverse. Ten forward/inverse pairs run in place per trial. Larger native sizes and the optional SINGLE_PRECISION case remain unmeasured.

The historical Julia port used ComplexF32 cuFFT and checked only eight round trips. That source is archived in the CUDA.jl child as `original_main.jl`. The current reference is a genuine custom Float64 radix8 port. The old implementation is a different precision/library/validation scope; it is not being declared mathematically incorrect as a Float32 FFT.

## Numerical evidence

Every trial checks all65536 complex forward values and all65536 round-trip values:262144 Float64 scalar components, absolute1e-6 matching the original default-double tolerance. An independent direct Float64 DFT checks the full spectrum, and the native original radix8 CPU reference is also checked against that DFT. Parseval relative error must be below1e-10. Forward checks prevent an incorrect invertible permutation from passing solely on round trips. Observed native/Julia maximum component difference is2.42e-13; all Julia variants agree exactly.

The forward spectrum is captured by an additional forward validation launch after each timed pair series. That launch, DFT/energy checks, copies and resets are outside timing. All four models use identical input fixtures and this same validation/timing boundary. Five warm in-process trials are retained; performance is the mean of trials2–5.

## Mapping and compiler feedback

Native mutable eight-value arrays become immutable ComplexF64 tuples. CUDA.jl and KA use4608B static shared/local storage. JACC explicitly requests4608B dynamic shared storage and stages a576-double zero template, adding global reads and one initial barrier. Resource/code-generation effects have not been profiled; no register, spill, bandwidth or bottleneck cause is inferred. This cache-resident case cannot establish large-image or DRAM scaling. FP64 complex/transcendental arithmetic requires an appropriate mixed-operation performance model rather than an FP32 peak ceiling.

KA candidate1 failed device-IR validation at a capturing `ntuple` initializer (`jl_f_tuple` and dynamic helper calls). Candidate2 replaced it with eight explicit tuple loads, preserving arithmetic/layout, and passed. KA has two program candidates/checks and one compiler-feedback repair. JACC's first draft intentionally reused that repair and passed its first check; this is sequential learning, not an independent framework-effort experiment. Current CUDA.jl reference passed its first candidate/check.

## Effort and provenance

This is the first prospectively metered benchmark for this worker. Separate begin/end tool rounds surround reference audit, KA translation, JACC translation and annotation. The broad workflow span overlaps those phases and must not be added to them. Consumed usage includes repeated/cached context, reasoning, coordination and shared helper work; it is not exclusive generated-source cost or monetary billing. Partial saved-source character/4 proxies remain separate. Marker alignment lag is reported by the meter. Historical CUDA.jl generation cost is unknown.

Scientific intent is a generic batched spectral transform. No application signal, physical model, imaging modality or PDE provenance is identified in this source.

JACC retained times were137.592,63.569,65.941 and66.130us (CV0.435). The mean includes all four; no outlier was removed or replacement timing series taken. The cause is unmeasured. A shared annotation-writer syntax repair added no GPU runs and is recorded separately from translation failures.
