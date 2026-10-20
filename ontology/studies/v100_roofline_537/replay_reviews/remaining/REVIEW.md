# Remaining replay assessment

CPU-only full selected control/memory source review of remaining20models. This replaces prior incomplete triage, not historical primary evidence. State safety is distinct from replay tool support, actual scientific acceptance, output isolation, residency and wholeROI representativeness.

| Case | Models | Static state/progress verdict |
|---|---:|---|
|hungarian|4|safe|
|openmp|4|safe|
|prefetch|4|safe|
|streamUM|4|unknown|
|blas-gemmEx2|4|safe|

## hungarian

Host flags are updated/read only between kernels. Native RUN macro calls cudaDeviceSynchronize after every stage, then host branches on managed globals. Julia run does the same before reading/writing its four UnifiedMemory flags. No CPU responder is required while a kernel runs.

- Restore all mutated global/static device arrays and managed flags for each pass; native has __device__ symbol storage and __managed__ control globals, Julia has CuArray/managed pointers.
- Host-controlled iterative stage/state trajectory and full assignments/cost/slack/guards must still pass. First-per-name/geometry is not every adaptive iteration.
- Managed control-page residency and host solver overhead are not represented by kernel DRAM counters. This is not evidence of floating-point roofline applicability for predominantly integer combinatorial work.
- Separate staging concern: Julia study_common.jl uses source-derived historical auxiliary directory; Landlock will safely reject until a reviewed path-only copy overlay redirects it. Replay assessment does not clear output gates.

**Memory tier:** Device arrays plus small managed control pages; integer-kernel roofline applicability must be explicit.

**Method:** Conditional per-kernel replay; if symbol/managed flag restoration fails use separately reviewed application replay with full host controller and exact checks, not a fixed stage sequence.

## openmp

Native OpenMP worker allocates its own cudaMalloc buffer for a disjoint host segment, performs ordered copy/launch/copy/free; Julia worker owns its own CuArray and synchronizes before copying/freeing. No running kernel consumes another CPU worker response or shares its writable buffer.

- Restore in-place device data before extra passes; retain allocation lifetime until each worker completes all intercepted/replayed operations.
- Native may use legacy default stream whereas Julia workers have task stream semantics; per-kernel profiling/serialization does not preserve host-thread concurrency or launch contention.
- Original full262144element outputs over both repetitions and all1/2/4/8/16/32worker factors remain mandatory. One first kernel per geometry does not measure all worker copies.
- Application replay may have nondeterministic thread/kernel order; do not assume replay matching across host workers without auditing actual stream/context and invocation matching.

**Memory tier:** Device allocation arithmetic plus transfer/allocation costs outside sampled kernel; avoid whole ROI bottleneck inference.

**Method:** Conditional per-kernel component counters; timeline instrumentation or explicitly supported range/concurrency profiling for the whole host concurrency question.

## prefetch

Managed A/B are CPU-initialized before the launch loop and inspected after kernel completion, never modified by a live host responder. The kernel updates B+=A, so managed B must be restored before each additional pass.

- Static state/progress safety does NOT preserve managed page residency. CPU writes every page before timed trials; prefetch changes device placement before each source iteration. Profiler memory backup/restore and cache flushing can change this physical state.
- Current first invocation per name/geometry selects warmup kernels, not a timed CPU-reset/prepared state. Prefetch/no-prefetch modes share name/geometry: the selector does not distinguish them. Native warmup launches add with no prefetch; Julia launches add then optionally prefetches afterward.
- Do not use this raw sample to claim prefetch benefit, migration cost or whole timed ROI compute/memory bound. Device DRAM bytes omit host-link transfer/page-fault work; cache-control all is not a residency-reset protocol.
- Original seven-update full-array result check and no automatic CuArray adaptor prefetch policy remain mandatory.

**Memory tier:** Unified managed memory: device DRAM versus host-resident/migrating pages must be measured, not inferred from pointer type.

**Method:** Per-kernel replay only as explicitly labeled instrumented sample. For actual policy comparison use a reviewed timed-state selector plus application/single-pass strategy and validated host→device residency/fault/link instrumentation; retain original full workload, no silent prefetch/no-prefetch rewrite.

## streamUM

Full source proves each task owns separate managed A/x/result allocations and is assigned exactly once to CPU if n<100 or GPU otherwise. No kernel waits on live CPU mutation of its OWN buffers. However CPU workers can concurrently write other managed allocations while NCU saves/restores context memory; this review cannot establish active2022.4 replay handling of those concurrent host writes.

- Do not call this a proven source race or framework limitation: logical task buffers do not alias and original independent full-result checks passed. Unknown concerns profiler memory capture and concurrency fidelity.
- Native OpenMP dynamic scheduling and Julia atomic task dispatch can change worker/stream assignments. Application replay is not automatically safe matching when assignment varies.
- GPU tasks n100/101/127/257/511 have different sizes/states; first-per-geometry may merge distinct task sizes. CPU-only small-task GEMVs produce no GPU arithmetic counters.
- Managed residency and concurrent CPU/GPU memory traffic are a mixed-tier workload; serialized kernel DRAM roofline cannot classify the whole dynamic scheduler.

**Memory tier:** Per-task managed buffers distributed between CPU/GPU with dynamic scheduling; actual residency/link/CPU work must be retained.

**Method:** Hold automatic multi-pass kernel replay approval. Establish active-tool capture/restore behavior with source-bound full-output evidence and reviewed bounded instrumentation, or use supported concurrency-aware range/application-range method with actual task/stream matching. Any frozen dispatch or GPU-only subset is a different named diagnostic, never original fullcontract.

## blas-gemmEx2

All models invoke the same explicit cuBLASLt pipeline with ordered default-stream matmul, input prefetch plus completion before each path, device workspace, and context completion before capture/destruction. No live host response or concurrent host update of active operands exists. KA/JACC are vendor-library integration, not authored alternative matmul kernels.

- Restore all vendor-written C and workspace state; full27path numerical/input/workspace-guard checks must still pass. Opaque kernel internals are not source-proven by this review, so profile/tool warnings remain binding.
- Selected operands are managed but explicitly prefetched to GPU and synchronized. Device DRAM interpretation can be component-valid only with residency evidence; pointer type alone does not prove migration or host memory access.
- Descriptor setup, heuristic search, synchronization and destruction are part of primary timing but are not GPU kernel counters. Different opaque algorithms/precisions/shape modes may share mangled kernel+geometry; first sample does not cover all27paths.
- Native corrected-reference arithmetic/compute-type qualification and exact loaded12.0.2.224 cuBLASLt identity must remain explicit. Do not attribute vendor algorithm behavior to KA/JACC frontend kernels.

**Memory tier:** Managed operands prefetched to device plus device workspace; allocation placement/migration must be verified independently for fullROI claims.

**Method:** Conditional per-vendor-kernel replay counters with full API pipeline checks; source-bound path/precision selectors and algorithm/counter provenance before any cross-model comparison. Application replay if vendor memory restoration fails, keeping same library/algorithm policy.

## Required distinctions

- Mapped host zero-copy (prior review) transfers operands/results over host link/system memory; low GPU DRAM bytes do not imply compute-bound. Its idempotent state is replayable but the memory tier remains different.
- Managed memory is an allocation mechanism, not a measured physical residency. Explicit prefetch, CPU touches, page faults and profiler save/restore can move pages. Separate resident device-kernel counters from end-to-end migration/prefetch benefits.
- Application replay reruns CPU work, but dynamic host-thread/stream assignment can break matching. No application/range mode is automatically authorized or presumed available in2022.4.
- Independent full scientific/input/output/guard checks and profiler errors must remain acceptance requirements. Landlock protects source/history writes, not numerical fidelity, scheduling or page-placement equivalence.
- Output-isolation gates are independent: Hungarian Julia auxiliary source-relative writes were observed statically and require a reviewed path overlay even though host-progress replay safety is conditional-safe.

All requested20slots have exact source/config/result/plan bindings; no GPU/compiler/target runs or source/shared metadata/Git edits. No missing profiler observation is recorded as zero or success.
