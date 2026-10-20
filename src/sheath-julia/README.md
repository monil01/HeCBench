# Sheath: a scoped electrostatic particle-in-cell study

## What is implemented

All three Julia versions implement the source's three GPU operators: FP32 cloud-in-cell (CIC) deposition, FP64 particle push, and FP64 half-step velocity rewind. Julia also owns the complete CPU density scaling, charge calculation, successive-over-relaxation (SOR) Poisson solve, electric-field differentiation, time loop, diagnostics and grid output. The C ABI provides **independent verification only**, not an application or GPU fallback.

The source uses a 100-cell, one-dimensional domain, grounded Dirichlet potential boundaries, absorbing particle walls, positive ions of mass16 AMU, and electrons. All versions preserve the source24-byte particle layout, including positions/velocities at offsets0/8 and the alive flag at16. Seven padding bytes are retained in raw storage but are not scientific output values. CUDA.jl uses its own atomics; KA uses public Atomix atomics; JACC uses its exported atomic macro. All three are `framework_device_kernel`, with no private CUDA device extensions. CUDA remains the recorded backend.

## Named cases and limits

| Case | Particles per species | Steps | Initial state |
|---|---:|---:|---|
| Original-size hot | 500,000 | 1 | Exact source initialized positions/thermal velocities |
| Smaller hot | 4,099 | 25 | Same source sampler and physics, named particle-weight scaling |
| Cold neutral control | 257 | 1,000 | Explicit zero-velocity, neutral positive control |

The cold control is not hot-plasma evolution. These cases do not jointly certify the original500,000-particle/species,1,000-step CLI, sheath formation, discretization convergence, energy conservation or application accuracy. The source's12-uniform velocity sampler is an approximate thermal sampler, not a validated physical distribution. A CPU witness compiled with the **verbatim original RNG helper bodies** confirms all2,000,000 initialized doubles in the original-size fixture exactly.

The original source is byte-preserved in `original_native/`. The native harness includes its unchanged GPU kernels and CPU charge/SOR/field functions. The original `main` is not executed: parameter scope, guarded allocation/lifetime, initialization observation and timing instrumentation belong to this named harness. Its original scalar-delete array footer is consequently not exercised or certified. No observed original-CLI runtime-failure claim is made.

## Numerical contract and the preserved revision

Every trial retains6,683,535 Float64 primary fields, including an exact3,026,145-field external-input prefix. It also retains complete unscaled particle trajectories, densities, charge/potential/field arrays, actual GPU field inputs and physical allocation bytes at every recorded stage. Every model receives five checked trials after a full untimed warmup.

Independent C++ witnesses include no CUDA or original application code. They check every CIC deposition using positive-sum/weight-rounding intervals, charge and integrated alive-particle mass, a long-double Thomas solution and Poisson residual, field differentiation, leapfrog motion and wall-loss flags. The CPU audit replays all retained stages and joins their actual bytes to the primary file. Each trial has15 guarded physical allocations;7,680 unique guard bytes are rechecked over snapshots, totaling2,641,920 actual guard-byte observations per trial. The complete five-trial replay checks13,209,600 guard bytes per model.

The first declared dual-reference criterion used thermal-scale potential units and mixed absolute/relative tolerances of1e-4. Native repeats rejected38–55 potential values per trial in the original-size case, despite passing the independent local mathematical checks; no other component failed that repeat comparison. The first criterion and its raw series remain rejected and unchanged.

A **separately named, post-observation revised contract** changes only the original-size case's potential normalization. `derive_neutral_phi_budget.py` reads source/input data, not measured outputs. It propagates count-specific FP32 CIC accumulation and density rounding through charge cancellation, the discrete Dirichlet Green function and the SOR residual stopping rule. Both field solves in this one-step case use identical original particle positions: rewind changes velocity only. The resulting per-node bound determines an outward-rounded normalization asset. All other component units, comparisons, Boolean flags and independent mathematical gates remain unchanged. Fresh native and target series use this contract; old timings are not reused across the change.

The maximum formal cross-model potential enclosure is5.398V, deliberately conservative compared with the largest observed raw difference of0.000667V. This enclosure is **not a physical-accuracy claim**. The independent solution check against each model's actual charge density remains tight. Initial external inputs are identical; endogenous fields can differ through atomic rounding. Complete physical values remain available separately, and `raw_output_identity_claim:false` does not assert literal padding or normalized-encoding identity.

The primary comparison uses the declared mixed absolute-plus-relative rule; its passing result does not imply that maximum relative error near zero is below1e-4. Reported per-trial error metrics are null when uncomputed; the central verifier separately computes complete comparison errors.

## Timing

The reported seconds sum the three named main loops, including CPU field work, resets, transfers, GPU motion, source diagnostics and the step1,000 grid write. Independent checks and full captures pause the timer. An explicit context-completion call follows field H2D transfer inside the field timer and follows particle pushes. CUDA's synchronous pageable H2D API need not complete final device DMA before returning, so the observer boundary requires an explicit completion protocol. See the [CUDA12.0 synchronization rules](https://docs.nvidia.com/cuda/archive/12.0.0/cuda-runtime-api/api-sync-behavior.html).

The prior field-handoff series remains preserved. A fresh symmetric series implements the completion correction, with no scientific-source change. Observer pauses, added context calls, runtime-versus-driver APIs and the cold control are timing covariates. This is neither the untouched CLI time nor GPU-only performance. Per-case full-ROI timings were not recorded and are not reconstructed from the aggregate.

| Model | Mean of checked trials2–5 (s) |
|---|---:|
| Native CUDA | 0.16224227625 |
| CUDA.jl | 0.16596079300 |
| KA | 0.17405038125 |
| JACC | 0.19727017575 |

No profiler counters or controlled ablation establish a bottleneck, roofline position or causal framework penalty.

## Effort and reproduction

Each target's first GPU program and all three mathematical bodies passed. CUDA.jl had one standalone host-parser whitespace repair before GPU submission. Native preparation used three build/process series: original strict criterion, revised potential criterion, then explicit field completion. These are not three different GPU mathematical algorithms. Unknown backend compiler invocation totals remain unknown.

Actual context-inclusive translation activity counters are CUDA.jl5,046,177; KA3,072,494; JACC2,704,402. They include cached/repeated context, coordination and verification waiting. CUDA.jl overlaps a shared completion-repair window and is excluded from clean cost calibration. Overlapping windows are not added. The native-readiness END accompanied a short final CPU review command, so possible tail undercoverage is disclosed without altering raw markers. These are neither billed/exclusive coding costs nor reconstructed historical consumption.

`essential_assets_for_commit.json` lists four required binaries totaling16,140,236 bytes: three particle fixtures and the808-byte potential-normalization asset. No raw run files, executables, shared libraries or generated grid files belong in Git.

```
g++ -O2 -ffp-contract=off -std=c++17 src/sheath-julia/prepare_initial_particles.cpp -o /tmp/sheath_fixture
/tmp/sheath_fixture src/sheath-julia/data
python3 src/sheath-julia/derive_neutral_phi_budget.py
python3 tools/verify_portable_julia.py --config src/sheath-julia/study_config_native_v3.json
```

Use the installed CUDA/Julia toolchains, pinned project environments and NumPy for derivation/audit. The central runner owns the GPU6 lock. Subsequent target configs reuse source-bound reference receipts; update their reuse paths when reproducing fresh series elsewhere. `audit_complete_capture.py RESULT MODEL CHECKER_SO` replays complete raw scientific data on CPU. The original-size source initialization witness and full proof/usage/ontology packets are preserved alongside the implementation.
