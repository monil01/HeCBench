# EasyWave: genuine shallow-water propagation and adaptive control

## Scientific operator and scope

The native program advances a linear shallow-water tsunami model through four GPU stages: surface-height mass balance with positive maxima/arrival diagnostics; serial radiation-height boundaries; interior staggered momentum; serial boundary momentum and active-area growth. Four GPU-computed bounds are copied to the host and determine subsequent launch grids. This dependent control makes each full Julia pipeline `hybrid_host_control`; its device component remains `framework_device_kernel`.

The named study uses 23 longitude × 19 latitude nodes, 12 Float32 fields/node, Int32 bounds, and eight steps. A signed wave over a dry island starts with a restricted rectangle and grows to bounds [2,2,22,18]. A second complete fixture is exactly resting water. Coefficients describe nondimensional g=1, depth=4, dx=dy=1, dt=1/32, and constant Cartesian metric R6=1. Source stage bodies and boundary order are unchanged. Earthquake/Okada generation, geographic grid/coefficient preparation, output files/POIs, and the original 120-minute geographic simulation are excluded. This is propagation-operator validation, not tsunami forecasting or inundation-quality certification.

## Numerical evidence

Independent Float64 vectorized staggered mass/momentum evaluation uses `hypot` for boundary norms. It explicitly adopts the source boundary/area policy; it is not an independent proof that every source corner condition is physically complete. Complementary saved-state checks verify resting equilibrium exactly, interior discrete mass/momentum balances, monotone positive maxima and bounds, exact first-arrival times, and unchanged depth/topography/coefficient fields. Threshold-clipped cells are excluded from the unmodified mass-balance residual because clipping deliberately changes that relation.

Each of the 20 artifacts has 10,730 exact prefix values, 346,168 checked output values, and 1,280 exact guards. The prefix contains actual initial node/coefficient readbacks and parameter descriptors, losslessly exported to Float64. Outputs include the actual timed final complete state plus all four stages of each of eight steps from a separate **untimed** identical replay, followed by exact adaptive-bound traces. Timed final states equal final replay states. All current source/fixture bindings are independently hash-checked.

Predeclared tolerance is 1e-7 + 1e-5 × |oracle|, unchanged after results. Maximum independent-oracle absolute error is 8.83636e-10; maximum normalized budget use is about 0.005616. Relative error is actually computed using max(|oracle|,eps(Float64)), not an uncomputed zero. All GPU implementations happen to agree numerically exactly on every serialized output here, a fixture observation rather than a general bitwise arithmetic guarantee. Input prefix bytes, arrival fields, bounds, and guards are exact. `saved_output_audit.json` retains every per-trial invariant/error result.

## Implementations and source limitations

All paths retain two 16×16 interior kernels and two single-thread boundary/adaptive kernels with the same latitude-fastest AoS layout. KA uses 2D GlobalNTuple indexing; JACC uses 2D parallel_for and serial callbacks, explicitly zero dynamic shared memory. No foreign device intrinsics or library substitution are introduced.

The original bottom-corner height predicate tests Jmin rather than Jmax; overlapping corner momentum updates also retain original order. These policies are not silently corrected. A separate source host-coefficient branch assigns R2 at the final latitude; geographic preprocessing is excluded rather than certified by these fixtures. These are static review caveats, **not invented observed numerical failures** of the selected defined propagation contract. Hmax is the source positive maximum, not maximum absolute wave height.

## Compiler loop, effort, and timing

The historical Julia file was an unrelated sinusoidal generator. It is preserved in `reference_original`; current CUDA.jl preparation proactively restores the full four-stage algorithm. One reference device-program candidate with four kernel definitions passed in its first actual compiler/execution evaluation; no feedback repair was needed. Native build count is one. Two CPU fixture-generator invocations occurred before measurement, the second adding actual timed-final and separate timed/replay guard captures to the declared contract.

KA and JACC each used one authored program, four kernels, and one actual device-JIT/execution check. KA first had a runner configuration-reuse rejection because a shallow-copied command list accidentally changed the reference command. It was rejected **before GPU/JIT**, preserved under `reference_history/ka_setup_attempt01`, and fixed without changing the candidate device body. Thus KA has two runner setup invocations, one setup repair, one actual compiler check, and no compiler/numerical repair. Stages, cells, steps and trials are not attempts; internal compiler jobs are unknown. Shared restoration/oracle work is separate from target frontend translation.

Both fixtures are warmed before five checked in-process outer trials. The ROI sums two eight-step, four-stage propagation pipelines including the four bounds H2D/D2H copies each step. Reset/copies, full-state snapshots, replay, checks and guards are outside the ROI. Mean excludes trial1 and averages trials2–5, preserving all samples. This is a **named instrumented host pipeline**, not kernel-only time or the original fullCLI timer.

Host allocation is a measured covariate: native adapter bounds readback creates one-element std::vectors, whereas the original CLI directly copies into stack integers. Julia uses preallocated host bounds buffers but constructs two empty trace vectors in pipeline(false). Launch wrappers and host allocation choices are included and differ. These timings cannot establish causal pure-frontend or abstraction-only performance differences. They remain immutable; no fastest selection or cleanup retiming occurred. Small-grid results have no bandwidth/roofline attribution without counters.

Mean summed pipeline time, milliseconds:

| Native CUDA | CUDA.jl | KA | JACC |
|---:|---:|---:|---:|
|2.7741155 | 3.1248735 | 3.09221075 | 4.36758475|

Pinned Julia1.12.6/CUDA.jl6.2.1/KA0.9.45/JACC1.4.0/localCUDA12.0 on V100 sm70. Essential CPU fixtures/oracles `study_case{1,2}_f32.bin` and `oracle_case{1,2}_f64.bin` have generator/hash metadata. Actual GPU captures remain local only. Historical attempts/tokens are unknown. Prospective counters include context/cache/coordination; shared workflow overlaps target phases, and JACC includes shared-queue/deadlock coordination. They are not exclusive billed translation costs.

Final bundle: `.porting-state/portable_julia/runs/20261018T063138630151Z_easyWave/result.json`; executable declaration: `study_config.json`. Coordinator owns separate verified-variant commits and paper integration.
