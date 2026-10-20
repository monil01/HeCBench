# MF-SGD observations (RQ1–5)

## RQ1 — Fidelity and evidence strength

All four implementations pass the predeclared **property contract**, with complete raw models and predictions retained. This is not a cross-process numerical model identity claim. The original sized RNG initializer overwrote 244 logical guard words; a minimal owned bounds correction preserved valid states and guards. Original feature initialization was nonrepeatable even with identical actual seeded states. Fixed-input native Hogwild training also changed 329,777 of 336,000 factor words across repeats (maximum 0.1866455), so raw model equality would be an invalid general gate.

The complete source one-partition SGD update runs the real MovieLens 80k/20k input, 30 epochs, rank 128, and 12/128-worker cases. Other partition/prefetch modes and Netflix/whole-CLI performance are unmeasured. Every model retains 120,320 actual GPU guard bytes and 30,720 host guard bytes over five trials, with exact actual input/selector joins. Independent checks cover 6.72 million Half conversions/model transforms and 1 million full train/test predictions per model. The wide-exponent 12-warp operator checks 3,072 exact-rational gradient components per trial under a derived FP32/FP16 bound and 62,720 unchanged components bit-exactly. Property gates do not prove unique trajectories, serializability, convergence or recommendation accuracy.

## RQ2 — Feedback, attempts and tokens

CUDA.jl used three source evaluations; KA and JACC each passed their first evaluation using the validated shared body/harness. The initial CUDA.jl generic-Int64 compile error and subsequent runtime InexactError came from translating CUDA C lane 0 directly to the one-based Julia indexed-shuffle API. The valid lane-1 control restored the ordinary Int64 helper; no framework unsupported claim is justified. Failure stages and revised cause are both preserved.

Context-inclusive observed translation windows: CUDA.jl 9,121,990; KA 2,688,960; JACC 2,801,849 tokens. Native readiness used 11,177,360 separately; annotations are separate and verification windows overlap translation rather than add cost. CUDA.jl creates the shared harness; later frontends inherit it and the API repair. These are ordered workflow activity counts, not independent language generation or billed cost.

## RQ3 — Execution/performance

| Native CUDA | CUDA.jl | KA | JACC |
|---:|---:|---:|---:|
| 0.132355 s | 0.138979 s | 0.143079 s | 0.142172 s |

Means use only trials 2–5 after full warmup. Each value sums the two complete source SGD-update functions (30 epochs, 12 and 128 workers), including allocation, rates, transfers, persistent updates, conversions and destruction. Eight timed GPU calls perform 4,899,840 rating updates per trial. Guard allocation/context fences/counters/selector tracing are explicit covariates. Raw capture is excluded after required completion. Model construction and postscale/unshuffle/predictions lie outside this source ROI. Conditional reference, allocator policy, fixed model order and a single process limit causal/performance generalization. Native printed self-timings include raw capture and are not the study timings.

## RQ4 — Mechanisms and ontology

Source-reviewed traits include FP16 storage with FP32 arithmetic, rank-128 four-chunk warp reductions, cached scalar loads, Int64 selector/modulo control and irregular Hogwild read-modify-write. CUDA.jl uses its own exported CUDA APIs; KA and JACC own their entries/dispatch but use explicit CUDA cache/shuffle helpers. Native legacy allocation and Julia pooled allocation are part of the scoped pipeline costs. No hardware-counter roofline, occupancy, cache or bottleneck attribution is established here.

## RQ5 — Scoped target selection

All three Julia frontends express the genuine named parallel learning pipeline and satisfy independently checked operator/property criteria. The current KA/JACC versions retain CUDA-specific helpers for faithful warp/cache mapping. This case supports configured feasibility and measured costs, with no generic cross-vendor portability or universal best-target claim.
