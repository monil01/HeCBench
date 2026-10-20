# LSQT / GPUQT observations (RQ1–5)

## RQ1 — Fidelity

All four implementations pass five complete checked trials on the predeclared five-model scope: the two original disordered-lattice solver parameter sets with bounded sizes, plus lattice/complex-ring/spin controls. Each trial retains 56,466,856 full-precision values, an exact 769,639-value actual-input prefix, complete physical allocations and actual guards. CUDA.jl/KA/JACC primary artifacts are bit-identical; their native maximum absolute difference is 2.3283064e-10 within the unchanged componentwise criterion. Independent long-double/Taylor witnesses check 49,880 values per model over all trials and appropriate large final norms. All 98,884 directed bond-current finite-difference checks pass their independently derived allowance. No original four-million-site/physical convergence certificate is claimed.

## RQ2 — Feedback and effort

CUDA.jl required a host parser repair and materializing a host reinterpretation before GPU transfer. Its first actual fourteen-body GPU candidate passes all trials; KA and JACC each pass their first whole GPU process. Low-level compiler invocation counts remain unknown. Actual context-inclusive declared activity windows are CUDA.jl 3,630,703; KA 1,777,060; JACC 1,588,966 tokens. Clean cost calibration excludes these respective windows for shared host/capture harness creation, another model's same-benchmark annotation, and shared scientific-witness creation. Nested verification windows are not additional tokens.

## RQ3 — Solver timings

| Native CUDA | CUDA.jl | KA | JACC |
|---:|---:|---:|---:|
| 0.819455 s | 1.101236 s | 1.343607 s | 3.376585 s |

Means use trials 2–5 after full warmup, with 171,073 timed launches per trial. Regions include source host recurrences/Bessel/spectral sums, allocation/destruction/transfers, GPU work and ordinary text output; explicit lifetime completion, enlarged guarded allocations and scalar/counter bookkeeping are covariates. Model construction and completed-work raw capture are excluded. These are bounded source solver-function timings, excluded from untouched-native whole-CLI aggregates, and cannot isolate launch/device lowering or original four-million-site performance.

## RQ4 — Mechanisms and ontology

This is atomistic solid-state electronic quantum transport with sparse FP64 complex Hamiltonian applications, Chebyshev recurrences and two-stage reductions, followed by host spectral sums. All models own fourteen device entries and six host solver algorithms. KA owns launch/index/local memory, while two ordered reductions use CUDA CTA/warp primitives. JACC owns dispatch/index/barriers but uses CUDA static scratch/warp primitives for those reductions. These backend dependencies are explicit. No profiling or controlled ablation establishes why the complete solver timings differ.

## RQ5 — Selection

The configured full quantum solver is expressible in all three Julia models, including host science, not merely a surrogate sparse matvec. Source warp ordering requires explicit CUDA-backend helpers for the KA/JACC reductions. This case supports scoped feasibility and observed solver costs, without universal target-selection, physical accuracy or cross-vendor performance claims.
