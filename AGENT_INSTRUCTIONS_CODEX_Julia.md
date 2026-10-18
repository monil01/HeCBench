# HeCBench Julia Porting — Agent Instructions (Codex)

All fixture and oracle reads must fail on missing files, short reads, unexpected dtype/size or parse errors. Declare native runtime assets explicitly when the verifier runs in a temporary working directory. A default-initialized input and a default-initialized oracle can falsely agree; a self-check alone is insufficient. Preserve such adapter failures and require exact actual-device input readbacks and full independent output checks before acceptance. Record missing/incorrect adapter inputs as infrastructure effort, separately from authored GPU candidates.


Within one verifier process, an already computed full numerical comparison may be reused for an identical ordered artifact-hash pair only after rehashing both actual saved files again and matching dtype, count, shape, byte order, tolerances, input-prefix count and agreement kind. Recheck the exact input prefix for each trial. Record `comparison_computation`; never reuse a GPU execution, self-check verdict, source/input provenance check or timing sample through this CPU-computation cache. All five independently executed trial artifacts remain required.

Use JSON `null` for an uncomputed numerical error metric, with a scope note; do not use zero as an unknown value. If archived rows already contain such a placeholder, preserve every row and add a hash-bound `reported_metric_definition_erratum` through `study_evidence.py`. Keep actual oracle absolute/conditioned errors and dual-reference comparison metrics separate. This descriptive correction cannot alter tolerances, sources, acceptance or timings and adds no GPU attempt.


An unchanged empty kernel (for example dispatch), or an original launch with a null destination that suppresses every observable numerical store (for example kernelLaunch), has no numerical outputs. For the latter retain the conditional store and by-value argument ABI, set `source_empty_kernel: false`, `source_no_observable_numerical_outputs: true` and `no_output_reason: original_null_destination`; source review must establish the original null path. Untimed nonnull/argument-echo diagnostics are separate evidence and cannot turn the primary launch into a numerical contract. Use `numerical_agreement_kind: no_numerical_outputs_execution_only` and `execution_only_verified`, retain source review/launch/completion and all five timing trials, and report these cases separately from numerical and property agreement. Never fabricate output arrays to obtain numerical acceptance. The runner's `execution_only_pass` is launch/completion evidence only; `variant_verified` refers to that explicitly named contract, not numerical correctness. No-output cases are excluded from numerical-contract checkpoint totals and artifact audits. A source runtime allocation/free benchmark with no device kernels uses `source_empty_kernel: false`, `source_no_device_kernels: true`, `no_numerical_outputs: true` and `no_output_reason: runtime_allocation_lifecycle`, with explicit API-status, nonnull-allocation and complete-lifecycle checks in every trial. Record zero authored kernels and runtime interoperation; do not invent an empty kernel or numerical output. Such lifecycle validation is execution-only and excludes numerical correctness and roofline claims.


For newly measured configurations, enable `bind_all_declared_dependencies: true` in every model specification and explicitly list executed includes, fixtures/oracles and runtime assets before the first measurement. This binds arbitrary extensions such as `.inc`, `.bin` and `.txt`; retain historical configs and their manual hash/audit evidence, and never imply that old runs had this additional automatic binding. Changing a reused model specification requires a fresh series or an explicitly reviewed provenance transition.

## Current authority: remaining-150 expansion (2026-10-18 UTC)

A nondeterministic property contract is not identical numerical output. Record `numerical_agreement_kind`, retain all raw states, and report property-equivalent cases separately from exact/tolerant numerical agreement in extension tables, receipts and checkpoints. Never serialize or alter a concurrent source algorithm merely to manufacture equal final values. If raw equality cannot be established under the original semantics, keep that requirement explicitly unmet while reporting the demonstrated property evidence. Library delegation is not an authored device-kernel port; record zero authored kernels and actual model-specific orchestration.

The user reopened translation: port the 150 existing CUDA.jl entrypoints lacking KA/JACC, verify native CUDA/CUDA.jl/KA/JACC agreement, and update the paper. Frozen worklist: `.porting-state/portable_julia/batch_150.json`; three disjoint worker queues are `batch_150_<owner>.json`. This adds 147 new names and revisits three previously hardware-limited cases, yielding 447 distinct names, not 450. The prior frozen300 results remain historical evidence and are not rerun automatically. These active instructions supersede completed-cohort stop notes below.

Use three workers with the existing owners; root alone commits shared/native changes and each verified KA/JACC version separately. Complete KA then JACC for each benchmark. Preserve original CUDA and CUDA.jl, fix references only with documented evidence, and retain conditional results separately. Exact contracts require exact agreement; floating-point contracts use justified predeclared tolerances. Capture complete outputs or explicitly justified property contracts, independent scientific checks, all five correctness verdicts and times; discard only first timing and average last four. Keep GPU6 serialized under the existing study lock; no use of other users' devices. Hardware limitations cannot be solved by fake outputs, changed operators or weakened acceptance.

Record actual attempts, diagnostics, numerical evidence, token activity windows (unknown historical consumption remains unknown), package/toolchain versions and per-model scientific/implementation ontology. Keep the full150 denominator and unstarted/blocked states visible; numerical agreement alone is not scientific validity. The paper must distinguish the original300 census from the147-name extension and report the447-name union without double-counting. Preserve source-backed scientific categories, RQs, compact readable figures, ten-page ACM body limit and local-only ignored research. Create active joint checkpoints; do not mark complete while repairable work remains.


## Current authority: completed frozen-300 assessments

All 300 benchmark names have reviewed scientific intent and separate CUDA.jl, KernelAbstractions and JACC records (900 model slots). The final evidence audit covers 297 matched names, 350 accepted configurations (348 primary and two explicitly corrected secondary configurations), and 7,000 checked trial artifacts, with zero hard audit issues. The three remaining names—`blas-fp4gemm`, `blas-fp8gemm` and `blas-mxfp8gemm`—have reviewed native hardware/library limitations; they are not successful ports or demonstrated framework limitations. Each model's conservative census separates 274 ordinary, 15 conditional, eight known-native-defect and three unavailable names. Do not describe this as 300 universally correct scientific applications.

Resume through the joint `2026-PASC-Hecbench-Julia-paper/research/checkpoints/LATEST.json`. A `complete_300_assessments_manuscript_review_ready` checkpoint supersedes historical active queues and completed-100 notes. Preserve the 100-name pilot as its own historical analysis. Do not automatically rerun or restart completed translations. GRRT's original configuration remains blocked; its independently checked numerical-method revision is a separate conditional secondary study. Preserve this distinction and MTF's corrected secondary algorithm in every aggregate.

The paper must explain scientific workload families and community reuse, organize experiments around research questions, and connect implementation-specific ontology to measured performance without asserting causation from descriptive correlations. Keep release versions separate from abstraction names. Report unknown historical attempts and partial/nonexclusive token accounting honestly. Author/publication metadata and an archival release remain submission tasks; the technical companion is a working document, not an approved PASC supplement. Keep research evidence local and ignored by the paper repository.


### Ready outcomes and conditional method sources

Before reporting a conditional or known-defect version ready, load its effective configuration and check the exact canonical `reference_provenance` field is nonempty, with source/correction/witness hashes. Alternate descriptive key names do not satisfy this gate. Use a hash-bound classification supplement for a missing descriptive field after measurement; preserve archived configs/results and all trial data.

A separately justified numerical-method repair retains its original blocked configuration and a named `corrected_reference_secondary` configuration. Record actual executed entry points and every source dependency, not only the canonical child-tree hash. Scoped nested model directories remain inside the benchmark's single Julia parent; the coordinator stages shared/native work and each model's actual scoped source separately, with explicit execution bindings in verification receipts. Never stage another model's new device source in the shared reference commit. Mathematical equivalence on an exact constraint manifold does not establish floating-point stability; preserve independent analytic/convergence controls and distinguish method changes from abstraction effects.


### Missing descriptive timing labels

A uniformly missing machine-readable timing label can be reviewed without rerunning when the original configuration already declared the region and source inspection proves it. Use `reuse_timing_region_metadata` only for explicit all-four-model reuse, twenty unchanged rows, original result/config hashes, hashed source evidence and reviewed boundaries for each model. Retain raw missing labels and all samples; the derived result adds a separate effective descriptor and review provenance. This cannot repair a different region, unit, completion boundary, input or numerical acceptance. Preserve the original nonmatched result and count annotation/tooling effort separately. The numerical audit independently checks the review hashes and original-row equality.


### Version-scoped capability conclusions

Bind every capability conclusion to the actual operation, compiler/toolkit, loaded library, generated target and device. A build rejection in one installed toolchain does not establish a universal hardware requirement: check another available toolchain when source/API evidence suggests a changed implementation. Preserve contrary successful probes and retract overgeneralized classifications explicitly. Capability execution alone is not full operator verification or a five-trial measurement. Explicit software storage/rounding that preserves every required low-precision intermediate is a semantic implementation choice; disclose it and generated arithmetic rather than silently replacing the contract with FP32, or inferring hardware instructions from a dtype name. Update per-model ontology and paper conclusions when new version-specific evidence changes a boundary.


### Native compatibility repairs and manuscript page accounting

Distinguish compile-only compatibility interventions (for example an exact zero literal selecting the intended FP16/BF16 constructor) from corrections to scientific mathematics or concurrency. Preserve unchanged source, rejected compiler diagnostics and the exact patch; conservatively classify a changed native reference as conditional and explain the intervention subtype. Never infer a scientific numerical defect from a build rejection.

For the PASC main manuscript, count every figure/table/appendix page toward the ten-page body limit, including floats deferred past the references. Flush all body floats before the bibliography when checking the limit. Move supporting detail into the separately identified working technical companion rather than reducing ACM fonts or margins; retain the main scientific conclusions, uncertainty and full-population denominators. The companion is not an approved conference supplement.



### Atomic operations and independent reduction checks

Read the actual atomic API expansion before translating reductions. An atomic store of `max(array[i], value)` can contain a non-atomic read and is not an atomic maximum. In the pinned Atomix API, use its modifying call form (`Atomix.@atomic max(array[i], value)`) for that operation; record exact package/source evidence. Check the complete selected index, tie rule, winning value and dependent state against an independent oracle, rather than checking only a plausible aggregate. Preserve failed candidates, all rejected captures and the original tolerance; classify an observed wrong result separately from a compiler rejection. Compiler success alone does not establish concurrency semantics.

### CPU and GPU co-execution regions

Use `mixed_host_device_pipeline` when a measured region includes CPU computation, device kernels and transfers, including CPU/GPU split sweeps. Record the authored GPU component separately with its own framework/hybrid class. Preserve all declared split settings and their discrete work partitions, CPU parallelism/compiler flags, asynchronous launch order, host work, copied prefix lengths and completion/readback semantics. Capture and check the complete answer at every split. Keep observation storage outside each measured interval and preallocate comparable buffers. An end-to-end sweep time is not GPU kernel latency; changing the CPU implementation or dispatch is a performance covariate. A kernel-only or selected-split follow-up needs its own declared configuration and cannot replace full-sweep evidence.

### Invariant microbenchmarks and executed work

A final checksum can be invariant to intermediate permutations, repeated swaps or other intended work. Validate every defined output, but do not infer that the intended loop or memory traffic executed from that agreement. Preserve vector width/alignment, unroll requests, synchronization and storage semantics; record generated resources and separate profiling evidence where available. Keep source-derived access counts separate from actual instructions/traffic and never label their quotient with time as measured bandwidth. Preserve original undefined readback/tail or type-reinterpretation checks as limitations; a scoped defined-kernel-output contract cannot certify the original whole-program checksum. Diagnostic observability changes or profiling runs have separate configurations and cannot replace the unchanged primary timing series.

### Rejected execution evidence

Retain structured records and binary artifacts from nonzero exits, timeouts and invalid-provenance executions under `rejected_trials`, with return code, source validity, raw diagnostics and capture errors. These records never establish accepted-trial completeness, reference comparisons, timing means or verification. Preserve nonfinite diagnostic values without corrupting the result JSON; keep original raw tokens and artifact bytes. A lost historical capture remains an explicit evidence gap. A later localization run has its own diagnostic configuration and scope; it cannot replace an earlier failed series or retroactively repair acceptance. Preserve failed candidates and original tolerances before any revised scientific contract.

### Loaded runtime library provenance

Record actual loaded library paths, resolved file hashes and runtime-reported versions for every vendor path. Compiler/link arguments alone do not prove which shared library the loader selected: RUNPATH, sonames and preloads can change it. A function address can identify an executable PLT stub rather than its implementation DSO. Preserve mistaken provenance as a hash-bound metadata correction; never relabel numerical success as an observed numerical failure. Keep unmatched library/compiler/runtime versions explicit performance covariates. A prescribed library-version control gets a separate diagnostic configuration and effort phase, fresh five trials for changed executions, exact dependency-checked reuse for unchanged models, and exclusion from primary aggregate counts before measurement.

### Exact sentinels and observed versus source-level hazards

Check sentinel and guard bytes exactly, separately from floating-point payload tolerances: a large sentinel with relative tolerance can admit corruption. When complete saved captures exist, a documented independent exact-guard audit can establish the declared contract without replacing their execution times. Disclose any weaker original self-check and retain a concrete checker counterexample; do not invent a GPU guard failure. If source analysis shows conflicting cooperative output stores but actual native trials pass, retain that contrary observation. A minimal sole-writer correction, when justified by the intended reduction and an independent oracle, remains a corrected-native intervention; a source-level hazard is not an observed numerical failure.

### Scientific fixture diversity and nonunique answers

Audit supplied data before treating repeated cases as independent evidence. Record identical weight files, repeated sequences, symmetry, zero/constant fields, conditioning and exact input diversity. Retain the supplied contract; add separately named varied fixtures when needed to distinguish plausible translation mistakes. Declare fixture choices and tolerances before collecting target timings, and state which original sizes or modes remain untested. Do not infer trained-model accuracy or scientific prediction quality from an operator check.

For nonunique scientific outputs such as graph colorings, preserve complete raw outputs and use an explicitly declared property-equivalence contract with an independent checker over every relevant node/edge. Record solution quality (for example, colors used), convergence iterations when observed, and rigorous problem-specific bounds where justified. Unknown internal iteration counts stay null. A normalized property vector is not exact raw-output equivalence; disclose algorithm and work differences in performance comparisons. Do not add asymmetric instrumentation to timed regions.

## Active expansion: all 300 benchmarks and scientific interpretation (2026-10-17)

**This section is the current authority.** It supersedes historical 100-pair completion/stop instructions, earlier feasibility-subset targets, and any instruction to commit `paper/research/`. The user has explicitly reopened translation and paper work, requested multiagents, and expanded coverage to the entire frozen 300-benchmark population. Read the other two canonical guides and the newest combined checkpoint. The previous 100 pairs/200 portable implementations/107 configurations remain valid evidence, not results for 300.

### Population, ownership, and completion

- Frozen population: `.porting-state/portable_julia/baseline_300.json`. Active census: `batch_300.json`; worker queues: `batch_300_{maps_ports,portable_infrastructure,stencil_ports}.json`. Do not replace difficult benchmarks with names outside the frozen 300. Re-examine the pilot's rejected candidates and preserve the rejection history.
- Target all 300 benchmark names in native CUDA, CUDA.jl, KA, and JACC: 600 portable versions in addition to the CUDA.jl references. Process one benchmark KA then JACC; three workers own disjoint queues and the coordinator owns shared tooling, source commits, ontology aggregation, and manuscript integration. All GPU timings remain serialized on GPU6 under `/tmp/hecbench-julia-study-gpu.lock`.
- Historical CUDA.jl PASS labels are not modern certification. Re-audit semantics, inputs, precision, reference accuracy, and timing region; preserve original source and account for reference repairs separately. Retain the completed pilot evidence where its contract/source/environment still match.
- Maintain the full denominator 300 in coverage tables, with separate model-specific counts: not started, in progress, verified, conditionally verified, failed, reference blocked, environment/hardware blocked, and evidence-backed feature unsupported. A stub, timeout without diagnosis, or unfinished port is not unsupported. Show library interoperability and scoped kernels separately from translated device kernels.
- Work continues through all queues and manuscript analysis. Never claim 300 successful benchmark pairs or 600 verified portable versions from target counts. If a genuine capability/hardware barrier persists after investigation, retain its minimal reproducer, logs, exact versions, and scoped outcome. Do not conceal it by replacing the benchmark or weakening equivalence.


**Scoped multi-GPU exception for the frozen ring benchmark:** the coordinator may use the otherwise idle physical devices5 and6 with `gpu: "5,6"` for a genuine two-device exchange. Retain the global GPU6 study lock, check device5 is idle before starting, and stop/defer if another workload owns it; never interfere with device0 or other users. Record both UUIDs, visibility/order, peer-copy semantics, synchronization, precision and scope. This is a separate two-GPU stratum, excluded from single-V100 aggregate rankings. KA/JACC host CUDA orchestration and zero-device-kernel interoperability must be labeled honestly. A one-device self-copy is diagnostic only and never satisfies multi-GPU correctness/performance coverage. Other benchmarks retain GPU6-only measurement, except the separately authorized P2P case below.

**Scoped P2P two-GPU exception:** `p2p` may also use physical devices5 and6 with `gpu: "5,6"`, under the same global GPU6 lock and device5 external-use guard. Recheck device5 before execution and around trials, and defer if an external workload appears. Require runtime peer-access capability in both directions, actual peer enablement, both UUIDs/visibility order, initialized full buffers, and full checks of both copy-phase and remote-kernel outputs. Preserve its authored remote-read device kernel and distinguish that framework-kernel component from CUDA host peer/copy orchestration; ring's zero-device-kernel classification does not apply to P2P. Record the source copy-only timed region, all synchronization and named iteration/buffer-size scope. A named 10-alternating-copy configuration does not certify the original 2,000-copy CLI workload. Keep P2P in the separate two-GPU stratum, excluded from single-V100 rankings; no one-device self-copy substitution or general relaxation for other benchmarks.


### Measured pipelines and component classes

Use `mixed_kernel_library_pipeline` when the timed computation combines authored device kernels with vendor computation calls. Record each component separately: framework kernels or device-intrinsic hybrids, vendor-library operations, their definitions/call counts, precision, stream binding and timing scope. Such a pipeline is neither a zero-authored-kernel interoperability case nor a framework-only kernel measurement. Do not attribute a whole-pipeline timing difference to one component without corresponding measurements. Keep measurement-repair commits/results separate from original translation attempts and costs; preserve original commit receipts when a corrected host protocol changes current source.

### Library streams and completion boundaries

For every library call, record the actual handle/generator stream, how it is set, the stream of dependent kernels/copies, and the completion API at the timed boundary. In the pinned CUDA.jl release, `CUDA.synchronize()` without arguments waits for the current Julia task stream. It does not establish completion of a raw cuBLAS/cuRAND handle on legacy stream zero. When native uses `cudaDeviceSynchronize`, use `CUDA.device_synchronize(blocking=true)` or a proven equivalent context-wide boundary. Check producer-to-consumer ordering before readback too; a host copy on another stream is not an ordering proof. Inspect high-level bindings before classifying them: they may already bind handles to the task stream.

Preserve numerically correct captures whose timing scope fails this audit, classify the measurement failure separately, and collect fresh five-trial series for affected implementations under a declared corrected protocol. Unchanged native evidence may be reused only with exact source/configuration/dependency checks. Do not infer an observed numerical error or a specific timing bias merely from the missing guarantee. Apply hash-bound exclusions through `tools/study_evidence.py`, using the `run_adjudications` schema, and assert the effective exclusion before replacement measurement; the mere existence of a metadata file is insufficient. Track shared measurement-repair effort separately from the original translations and retain their commit receipts.

### Attempt units and measurement instrumentation

Record program candidate revisions, process evaluations, frontend/macro checks, actual device-JIT checks, kernel definitions and dtype/shape specializations separately. A parser, namespace or kernel-macro rejection before device JIT is an observed frontend/API failure, not a completed device compilation. Preserve its candidate and diagnostics. A later source revision that reaches its first device JIT is still a later program candidate. Five correctness/timing trials and eight numerical cases are not five or eight translation attempts. Attribute shared host-driver or measurement repairs separately from portable-kernel compiler feedback.

Allocate diagnostic output buffers and initialize recording counters outside the timed region in every model. Keep the genuine scheduling, dispatch, library calls and required completion inside the declared region. If review finds an instrumentation mismatch after a correct series, preserve that series and its numerical evidence; record the mismatch before collecting a replacement, rerun all affected models with five trials, and exclude the superseded series from primary timing aggregates through a hash-bound supplement. A replacement is selected by the revised protocol, never by its speed.

### Scientific counterexamples and reference allocation repairs

For domain-facing case studies, connect a diagnosed semantic change to a concrete scientific consequence where a justified witness exists: for example, repeated Sobol coordinates alter a known two-dimensional integral even when the same-table self-check passes. Keep full output/reference checks alongside such witnesses. A single integrand, invariant or finite-prefix balance does not establish general convergence, statistical quality, or application validity. State which rejected shortcut the witness distinguishes and its limits; preserve contrary/negative findings.

A host allocation or initialization correction is still a native reference intervention even when device kernel bodies remain unchanged. If padding repairs a demonstrated inactive-lane out-of-bounds access, record the original access/allocation witness, old/new logical and physical sizes, defined padding values, and `native_reference_repaired:true` for the affected configuration. Keep a genuinely unchanged base variant separate when applicable. Exclude the repaired configuration from untouched-native aggregates. Distinguish source-proven invalid access from an actually observed sanitizer/runtime failure; do not invent failed checks.

**Reported error metrics:** an uncomputed metric is null/unknown, never a zero placeholder. Define each denominator and floor explicitly; a residual divided by an absolute tolerance is a scaled absolute error, not a relative error. If an archived adapter mislabeled a diagnostic, preserve raw source/trials and attach a hash-bound `reported_metric_definition_erratum` with actual saved-artifact recomputation. This contextual supplement cannot change execution, acceptance, timings, or repair failed checks. Central cross-reference errors and independent mathematical-oracle errors are different comparisons.

**Configuration strata and benchmark denominators:** a benchmark can have an untouched base configuration and a corrected optimized configuration (for example, meanshift). Report configuration-specific eligibility and distinct benchmark unions. When checkpoint stratum name sets overlap, list overlapping names and label the counts non-additive; do not sum them as a disjoint census or discard valid unchanged configurations. A per-model status table using one conservative outcome per benchmark must state that aggregation policy.

### Required all-300 manuscript deliverables and scientific questions

The main paper must use the frozen 300-name census as its population; retain the 100-name pilot only as a clearly identified historical cohort. Publish a compact scientific-family-by-implementation coverage panel and a failure-category-by-implementation panel for CUDA.jl, KA, and JACC. For each panel distinguish reviewed unknowns, pending work, diagnosed barriers, conditional scientific/reference outcomes, and successes; give the denominator and evidence coverage. A shared taxonomy vocabulary does not replace separate model-specific observations. Connect model-specific ontology changes to the same benchmark/configuration in the figure data rather than assigning one benchmark-level mapping to all three implementations.

For each scientific family, maintain a source-backed question, experiment, observation, limitation, and community implication. Questions should invite further investigation: for example, whether reduction order matters at the application's accuracy scale, whether synchronization or shared-memory initialization explains a layout-sensitive performance gap, or whether an operator check transfers to a long simulation horizon. Label post-observation explanations and proposed controls as exploratory; an unexecuted hypothesis is not a result. Prefer complete-field accuracy and measured mechanisms when available, preserving weaker evidence honestly. Show how a community member can select a comparable operator, reproduce its named accuracy/performance contract, locate the relevant abstraction limitation, and extend the experiment. Do not equate 300 operators with 300 validated applications or one GPU backend with demonstrated cross-vendor portability.

Keep the ACM main manuscript within the official PASC page budget. Preserve detailed case histories in a separate technical companion and local evidence rather than dropping their provenance or crowding the main narrative. Appendices in the main manuscript count toward its page budget; do not assume an external supplement is accepted without venue confirmation. A checkpoint must list pending manuscript panels, unresolved scientific questions, companion material, and the current body-page count alongside benchmark/ontology progress.

### Meaning of Julia version and separate implementation records

Here, implementation versions mean **CUDA.jl, KA, and JACC**. Also record the actual Julia language release, CUDA.jl/KA/JACC package versions, manifests, driver/toolkit, GPU, and source/configuration hashes independently. Package/model comparisons are not evidence of differences between Julia language releases. A claim about runtime release changes requires an explicit matched release experiment.

Use the key `(benchmark_id, configuration_id, implementation_model, environment_id, source_revision)` for every ontology, failure, effort, numerical, and performance record. Keep native CUDA as a separate reference model. Preserve evidence for prior revisions and interventions; never overwrite the pilot to make histories look uniform.

### Per-implementation failure taxonomy

Retain the stable B1–B4/R1–R3/N1–N3/S1–S2/P1/T1 taxonomy. Record each observed event independently for CUDA.jl, KA, and JACC, with event ID, phase (reference repair, translation, compile/JIT, runtime, numerical verification, measurement, tuning), source candidate/check/repair IDs, exact error, minimal trigger, root-cause evidence, resolution, final outcome, and source/log hashes. Use multiple linked events where one defect crosses phases; distinguish symptom from diagnosed cause. N3 tolerated drift and P1 documented compromises are observations, not failed attempts. Preserve explicit diagnostic-only and out-of-contract findings without counting them as failed translation attempts. Distinguish an unchanged native boundary policy from a corrected native numerical reference; agreement with that policy does not validate an idealized physical or mathematical boundary condition.

Produce per-model breakdowns and benchmark-by-model tables; report (a) failure events, (b) affected benchmark/model pairs, (c) repair rounds, and (d) unresolved final outcomes separately. Show denominators and unknown-field coverage. Historical CUDA.jl costs and failures may be unavailable: missing records remain unknown, not zero or reconstructed from the current source. Include representative domain-relevant failure-to-repair case studies, such as changed discretization, boundary handling, reductions, random-number semantics, mixed precision, device API limitations, or asynchronous timing.

### Ontology for each implementation and scientific workload

The benchmark ontology has two distinct layers: **scientific intent** and **observed implementation**. Keep the inferred base YAML with its provenance; add reviewed study records under `ontology/studies/julia_abstractions_300/`. Shared intent must not be copied into model-specific performance claims. Every one of the 300 names needs a domain record; every CUDA.jl/KA/JACC slot needs an implementation record, including explicit unknown/unimplemented slots.

Scientific intent includes primary/secondary application domains, scientific task, governing mathematical operation or numerical method, discretization/mesh/topology when applicable, conservation/invariant/statistical accuracy needs, input-size regimes, precision sensitivity, and a concrete community use case. Classify domain-neutral primitives and hardware microbenchmarks explicitly; do not invent application provenance from a benchmark name. Candidate families include PDE/continuum simulation, molecular/particle methods, signal/image processing, numerical linear algebra, stochastic/statistical computation, scientific machine learning, graph/irregular algorithms, discrete/cryptographic computation, and hardware/data-movement probes. Multiple domains are allowed; source-backed annotation and uncertainty are required.

Per-model implementation ontology records actual algorithm, work-item/data mapping, layout, precision, memory hierarchy and access pattern, synchronization, atomics/reductions, resource policy, launch geometry, library/backend extensions, host/device boundary, supported feature and fidelity status, and measured configuration. Distinguish intrinsic scientific requirements from chosen implementation details and inferred versus measured properties. Link native-to-model changes, numerical evidence, attempt/failure events, all trial times, and profile IDs. Unimplemented slots contain null traits/status, not guessed device behavior.

Use this to ask why abstractions suit scientific requirements, and which effects arise from changed algorithms, layout, resource use, or orchestration. Treat ontology/performance associations as descriptive until controlled interventions support a mechanism. Stratify by scientific family, computational motif, precision, problem size, and mapping fidelity. Use paired within-benchmark comparisons, show stratum sample sizes and missingness, avoid treating multi-configurations as independent benchmarks, and separate multiple-domain overlaps. A floating-point roofline is not meaningful for integer/cryptographic work; use measured bandwidth, latency, occupancy, or relevant operation measures instead.

### Verification, timing, effort, and commit requirements

The existing dual native-CUDA/CUDA.jl reference checks and five-trial protocol remain mandatory: identical input fixtures, tolerances and synchronized timed regions; correctness on all five trials; discard only trial1 time and average trials2–5. Keep raw samples/outliers, explicit warmup, reference agreement, full output or honestly weaker property evidence, and named scope restrictions. Add source-grounded scientific invariants when useful, without presenting an invariant alone as full numerical equivalence. Stochastic codes require an explicit identical-stream or distributional contract.

Record translation candidates, compiler checks, feedback repairs, runtime/numerical repairs, correctness, performance, and token usage separately per model and phase. Start usage boundaries before new work when available; shared orchestration remains shared. Missing exact consumption stays null and approximate records must state method and coverage. Never compare total costs using heterogeneous source-size proxies. Record shared mathematical helper origin (copied original versus newly authored), its source hashes and owning effort phase. Attribute new shared algorithm/oracle work once. KA/JACC activity windows are marginal costs of the ordered workflow conditioned on completed reference helpers and prior feedback; they are not independent cold-start translation costs or intrinsic abstraction costs. Expose reuse conditions alongside candidates, checks and token counts.

On a new root session, run `tools/register_translation_usage.py` to bind the current study streams; never subtract counters across different stream identities. Close stale incomplete segments with `--boundary abandon --note <reason>` and retain unknown cost. Prospective usage measurement is now available through `tools/capture_translation_usage.py`, using registered cumulative harness streams for this root and its three workers. Before each untouched benchmark, record a shared `benchmark_workflow` begin; capture CUDA.jl reference work and KA/JACC translation begin/end windows separately. A begin must occur in a separate tool/model round **before drafting that phase's code**; close in a separate subsequent tool/model round after the generation, tool result and verified or diagnosed outcome. Never place END in the same exec call as the measured code/metadata write; preserve and label any observed undercoverage. Do not backfill started work. Record exact usage-event timestamps, input/cached/output/reasoning counter deltas and marker lag; these are observed activity-window counts, not exclusive billed translation costs. Repeated context and coordination are included. Disclose interleaved benchmark work; overlapping workflow/subphase spans cannot be summed. Preserve earlier partial source proxies and unknown history alongside these prospective records, and report measurement-method coverage separately. Verify the identity of every profiled kernel; initialization kernels do not characterize the scientific operator. Profiling/filter repairs remain shared measurement effort, not target translation failures.

Only the coordinator stages source commits; preserve one separate commit per newly verified KA/JACC version and distinguish reference/infrastructure/tuning commits. Workers report ready result/config paths and do not stage another worker's files. Never commit `.porting-state`, research evidence, executables, measured output archives, or unrelated work by broad staging. Small essential generated binary input fixtures may be committed explicitly with their generator, metadata, hash and scope; these are source inputs, not build products or raw measured outputs.



**Input-dependent scientific and performance context:** attach measured or directly derived canonical-fixture descriptors to the configuration, with their derivation and source hash. Examples include graph degree/active-frontier distributions, sparsity, histogram entropy/concentration, active interaction counts, branch populations, solver iteration/accepted-step counts, and conditioning indicators. These describe the named input, not an intrinsic framework property. Use them to formulate matched experiments and stratified associations; a low-entropy histogram alone is not evidence of an atomic bottleneck, and a matrix spectrum alone is not a performance explanation. Preserve unknown descriptors rather than infer them from a benchmark name.

**Controlled performance explanations:** retain the primary framework implementation and every prescribed trial when evaluating a backend-specific diagnostic. Give controls a distinct configuration ID and inline `configuration_role: diagnostic_ablation`, exclude them from primary census/family/translation-cost rankings, and meter tuning/profiling separately. Reverify all five outputs and identical inputs/regions. A fixed requested shared-memory size does not fix generated registers, local-memory traffic, cache/shared configuration or theoretical launch limits; measure these before attributing a gap to one copy or allocation. Thread-local traffic does not by itself distinguish stack, spills or local arrays. Profile first-kernel/replay observations separately from unprofiled batch times; do not replace a primary result with a favorable diagnostic.

**Evidence corrections:** preserve immutable run/config bytes when correcting a prose count, scope interpretation or classification. Store a hash-bound `study_adjudications.json` supplement through `tools/study_evidence.py`, with explicit `input_contract_metadata_erratum` or `source_scope_adjudication` notes. These fields cannot change actual inputs, numerical acceptance, trial times, sources or launch behavior. State the measured value and source evidence; do not turn an untested input regime into a demonstrated failure.

**Exact input identity is separate from output tolerance:** future structured configurations with serialized input prefixes must declare top-level `input_prefix_count` (number of leading artifact elements). The runner compares those prefix bytes exactly before applying output tolerances; missing declarations make no automatic prefix claim. Keep an independent producer/source or actual-buffer audit because serialized-prefix equality alone cannot prove that a kernel used that data. Retain wrong-fixture failures, including one-ULP differences caused by `expf` versus double-precision generation, as shared instrumentation/input events. Do not silently relax input identity or count such repairs as target compiler attempts. Existing separately recorded byte audits remain evidence; adding this option does not retroactively certify old runs. Strengthening prefix validation on unchanged reused captures requires `reuse_prefix_revalidation_reason`; all saved references/targets are rechecked under the stricter rule without editing archived configurations. A target's own PASS never overrides a failed or missing comparison against either reference; all five explicit comparisons must pass for verified status and matched performance.

**Precision has an execution role:** separate storage and arithmetic in the timed device kernels from untimed GPU setup, host fixture generation, independent CPU/BigFloat oracles, and output-artifact encoding. An FP64 oracle does not make an FP32 kernel an FP64 workload; an untimed FP64 RNG stage does not make the timed integer rank kernels FP64 roofline work. Retain source-bound stage-specific precision and unknowns rather than infer device precision from shared helper annotations or serialized output dtype.

**Model-relative implementation class:** CUDA.jl device kernels using CUDA.jl's own atomics, shuffles, pointers or shared-memory API remain `framework_device_kernel`, with their CUDA-specific features recorded. KA/JACC kernels calling those CUDA.jl device APIs are `hybrid_device_intrinsics`; vendor-library interoperability is a separate class for any model. Explicit foreign device-library calls must state their ABI and backend dependency. Preserve class, mapping and resource differences per configuration when one benchmark mixes pure and hybrid modes; a benchmark-wide label must not overwrite configuration evidence.

**Agreement is not scientific acceptance:** complete saved captures and exact four-model equality do not promote a configuration whose independent scientific check is false. Keep it `reference_blocked` (or another diagnosed unresolved status), retain every failed check/artifact and explicit capture gaps, and exclude its times from matched scientific-performance aggregates. A diagnostic process may return zero solely to preserve `correct:false` records; process success and capture completeness are not numerical success. Investigate arithmetic, adaptive-step/event-location, discretization and observable sensitivity before diagnosing a framework limitation; an integrator local-error target does not by itself bound final observables. Never loosen the primary acceptance threshold to obtain a pass.

**Accuracy scales and scientific witnesses:** report implementation agreement, floating-point arithmetic error, discretization/model error, and iterative-solver solution error separately. An iterate-change stopping norm is not an absolute solution-error guarantee; a small equation residual also depends on conditioning. For force or derivative kernels, distinguish absolute errors from scale-normalized conservation or accuracy diagnostics. Keep the native/CUDA.jl port tolerances fixed across models. A revised auxiliary mathematical-check budget requires an independent precision/scale justification, the original failed check and artifacts, an explicit revision event, and disclosure that it was chosen after observing that check. Never reclassify that revision as a framework compiler failure or silently present it as a prospective threshold. Guard, boundary, cutoff, and singular-input probes establish the source's defined behavior; they do not establish that those inputs are physically admissible.

**Native numerical-reference repairs:** a native/CPU disagreement first needs a minimal witness and a source-grounded mathematical contract; do not assume either reference is correct. If a native defect is independently established, a corrected study kernel is authorized only with the original source and mismatch preserved, exact correction/source hashes, independent numerical evidence, and separate native-repair effort/failure records. Set `native_reference_repaired: true` and a `reference_provenance` object in the configuration. Classify results as conditionally verified against the corrected reference, not verified against untouched native CUDA; exclude corrected-native timings from untouched-native aggregate charts and show the conditional stratum separately. Do not silently change a physical model, regularization choice, tolerance, or scientific question to manufacture agreement.

**Reproduced native behavior with an established scientific defect:** if an independent witness establishes a semantic defect but the study intentionally preserves the original native behavior, set `native_behavior_known_scientific_defect: true`, `exclude_from_scientific_fidelity_aggregates: true`, and explicit witness/reference provenance. Report `verified_native_behavior_known_scientific_defect`, separately from both ordinary named verification and corrected-native conditional verification. Such evidence confirms reproduction of executable behavior, not the intended scientific algorithm. Keep its timings in a separately labeled diagnostic stratum; exclude it from faithful-operator aggregate charts and abstraction recommendations. Retain a justified correction as a separate repair task, with original outcomes preserved. If classification follows execution, use a classification-only `src/<bench>-julia/study_adjudications.json` supplement bound to the immutable result/config hashes and loaded by `tools/study_evidence.py`. Do not mutate archived configurations or rerun just to change a descriptive label; supplements cannot change inputs, commands, tolerances, trial data, or pass/fail results.

### PASC paper and persistent restart

The manuscript must engage domain scientists: begin with scientific decisions, why a 300-workload corpus is useful, and what transferable evidence a simulation or analysis developer gains. Present scientific workload categories and numerical requirements before implementation taxonomy and aggregate speedups. Explain limits of kernel proxies for whole applications and of one NVIDIA platform for portability. Do not promise community artifacts that are not actually distributable.

Keep RQ1–RQ5 but orient them to (1) scientific-family coverage and fidelity; (2) model-specific compiler/semantic failures and effort; (3) paired performance by family and scale; (4) scientific-intent and model-specific ontology mechanisms; and (5) actionable abstraction choices for scientific use cases. Show representative scientific case studies and open hypotheses, with readable compact figures, per-model taxonomy panels, coverage/unknown denominators, and evidence-backed recommendations. Keep the common four-language example and combined methodology diagram. Distinguish the census target from current completed results in abstract, tables, and conclusions.

`2026-PASC-Hecbench-Julia-paper/research/` is **local-only and ignored**. Keep data, scripts, observations, and joint checkpoints there for session restart; do not force-add them to the paper remote. Only LaTeX, references, snippets, final publication figures, generated LaTeX summaries, and manuscript documentation belong in Overleaf sync. Local research is not a published archive. Export a small checkpoint pointer/summary to the source repository without raw artifacts. Checkpoints record both repository HEADs, all300 status/model counts, ownership, versioned ontology/taxonomy schema, source and paper progress, running jobs, and exact next actions. Historical complete100 checkpoints do not stop this expansion. Census completion means all300 names and all900 implementation slots have a reviewed outcome, not300 passing pairs. An unavailable outcome requires evidence plus explicit `terminal_adjudication: true` and `repair_or_limitation_review_completed: true`; a repairable reference backlog or unfinished attempt cannot be marked terminal to close the study. Report unavailable and scientifically qualified strata separately. `checkpoint_portable_study.py --complete` enforces reviewed outcomes and a refreshed numerical audit; do not invoke it while work remains.

Coordinator priority: keep the 300-slot inventory, scientific classification, per-model evidence ledger, and checkpoint generator current while workers translate. Commit this protocol amendment before results depend on it.

---


> **Which file to read.** Two run-books live at the repo root:
> * `AGENT_INSTRUCTIONS_CODEX_Julia.md` — this file. Follow it when the
>   driver is OpenAI Codex (CLI or API).
> * `AGENT_INSTRUCTIONS_CLAUDE.md` — companion for Claude Code.
>
> This Codex run-book covers **Julia CUDA.jl, KernelAbstractions.jl (KA),
> and JACC.jl ports**. Do not create, repair, or extend Serial, Triton, Mojo, Rust, HIP,
> SYCL, or OpenMP ports unless the user explicitly changes the scope.

## Start here: resume translation and paper together

This file is the canonical entry point, including when the user refers to
"agent instruction codex julia.md". Always open these exact companion files:

1. [JULIA_CORRECTNESS_CHECK.md](JULIA_CORRECTNESS_CHECK.md): current numerical
   status, evidence strength, limitations, and verification instructions.
2. [paper_writing.md](paper_writing.md): concurrent PASC paper workflow,
   ontology/measurement links, section files, observations, and claims.
3. [LATEST.json](2026-PASC-Hecbench-Julia-paper/research/checkpoints/LATEST.json):
   pointer to the most recent fully saved combined checkpoint (including an
   in-flight checkpoint). Read its referenced
   checkpoint and any linked per-variant or in-flight checkpoints before work.

Set the repository root explicitly to the directory containing this file and
set `STATE=${PORTING_STATE_DIR:-<repo-root>/.porting-state}`. Resolve paths from
that root, not from the shell's incidental working directory. Check branch,
HEAD, dirty files, live source layout, database status, saved artifact paths,
and any running jobs against the checkpoint. Preserve unrelated work. A stale
checkpoint is evidence to reconcile, not authority to overwrite newer work.
Resume the named benchmark/variant and pending paper action before selecting
new work. Missing evidence stays unknown; do not invent results or rebuild the
whole experiment just because the conversation was lost.

**Current user-authorized extension (2026-10-16):** Add KA and JACC versions
of the frozen set of 300 CUDA.jl benchmarks with passing numerical records.
Use the nested source layout in §13.1, verify each against CUDA and CUDA.jl,
and record five-run timings, translation tokens, failures, and feature gaps.
Develop the paper in `2026-PASC-Hecbench-Julia-paper/` alongside translation.
**§13, §14, and paper_writing.md supersede older CUDA.jl-only restrictions,
flat source paths, manuscript destinations, missing-port worklists, and
performance-subset guidance.** Older examples remain historical guidance for
existing CUDA.jl ports; resolve their actual paths through §13.1.
This documentation update does not migrate sources or claim new ports/results.

This document is the run-book for the **next Codex-driven agent** that
continues the HeCBench Julia porting effort. Read it
end-to-end before doing anything else — the "why" for every step is
captured, so you can make judgement calls when reality diverges from
the recipe. After each completed Julia port, write a local context
checkpoint under `${STATE}/porting_logs/<bench>-julia/` before starting
another port. Do **not** ask the user to run a remote/platform context
compression task.

## Codex context hygiene — avoid remote compaction

Codex remote context compaction has been observed to fail with
`404 Not Found` from `/backend-api/codex/responses/compact`. Treat remote
compaction as unavailable for this project. The run-book must be executable
from local files plus git history, not from a long chat transcript.

Operational rules:

* Default to one Julia port per agent, but when the user explicitly asks for
  parallel agent work, split the remaining worklist across sub-agents. Start
  with the verified short-list (§3.2.2), then continue into Batch A/B/C
  candidates when that list is exhausted so the run can reach 50 more Julia
  ports. Each sub-agent must own disjoint `src/<bench>-julia/` directories
  and must still finish, verify, test, log, track, and checkpoint each port
  before moving to its next assigned port. The current batch objective is
  for each agent run to port at least 50 remaining benchmarks and at most 200
  benchmarks before moving to broader Batch C work.
* Before reading or pasting large outputs, write them to
  `${STATE}/logs/*.log` and summarize only the decisive lines in the chat.
* Do not paste full source files, full build logs, or full verification logs
  into the conversation unless a specific short excerpt is needed.
* After every successful port commit, write the local checkpoint described in
  §6. In a parallel batch, the coordinator may keep the session open while
  sub-agents work, but each completed benchmark still needs its own checkpoint
  before another benchmark is assigned to that same worker.
* If context is getting long before the port is done, pause implementation
  long enough to write
  `${STATE}/porting_logs/<bench>-julia/context_checkpoint_inflight.md` with
  current files changed, commands run, failing error, and the next concrete
  step. Continue from that file instead of requesting remote compaction.
* If a Codex UI or CLI offers remote compaction, do not trigger it manually
  for this workflow. Use local checkpoints and fresh sessions.

## 0. Codex restart scope — keep the existing ports

Every `src/*-julia/` directory that already exists on this branch may have
been produced by another agent. **Do not re-do it, do not delete it, and do
not overwrite its manifest.** Your job starts at the *worklist gap* — the
CUDA benchmarks that still have no Julia sibling.

Concretely:

* Read the "already done" lists produced by §3.1 and *skip* those
  benchmarks unless the user explicitly asks for a re-port.
* If you re-run the verification on an existing Claude-authored port
  (fine, encouraged as a checkpoint) and it fails, record the failure
  under `${STATE}/porting_logs/<bench>-julia/reverify_YYYY-MM-DD.md`
  and flag it — don't rewrite the port until the user says so.
* Every port you *do* produce gets `"driver": "codex"` in its
  `manifest.json` (§6). This is what lets §7's cost-study tables
  keep the two drivers in separate columns.
* After each completed Julia port, write a local checkpoint before starting
  the next port. Use
  `${STATE}/porting_logs/<bench>-julia/context_checkpoint_NN.md`.
  Summarize the commit, files changed, verification commands/results,
  manifest path, worklist updates, remaining unrelated worktree changes, and
  any pitfalls the next agent needs. Do not request remote context
  compression from the user.

**Existing CUDA.jl focus: numerical correctness against CUDA references.**
The KA/JACC extension follows §13 alongside this correction workflow. The broad translation tranche is
finished/triaged:
the live tree has 537 CUDA benchmark directories, 448 Julia benchmark
directories, 89 CUDA benchmarks without Julia directories, 91 local
deferred records, and 0 actionable non-deferred missing benchmarks. The
coverage database currently records 441 Julia `ok` results and 5 Julia
`run_fail` results (`aobench`, `kalman`, `lrn`, `mis`, `tsp`). Do not
continue broad translation work unless the user explicitly reopens a
specific deferred or failing benchmark. The active project objective is to
correct the failed numerical checks and close the gaps from the numerical
accuracy sweep in §11.


Existing non-Julia implementations may be used as reference material or
verification inputs when they already exist, but they are not targets for new
work in this pass.

The active paper destination is `2026-PASC-Hecbench-Julia-paper/`.
Follow [paper_writing.md](paper_writing.md), keep each section in its own TeX
file, and connect every empirical claim to versioned evidence. Existing draft
content is not the required narrative. The old `Hecbench-agent-paper/` paths
in historical examples below are not the active writing destination.

---

## 1. Goal in one paragraph

HeCBench ships kernels in CUDA / HIP / SYCL / OpenMP-target. The broad
Codex Julia (`CUDA.jl`) translation tranche is complete for this pass:
passing ports were committed, infeasible remaining CUDA benchmarks were
recorded as deferred, and no actionable non-deferred missing Julia
benchmarks remain. The active goal is now to audit numerical accuracy for
the existing Julia ports by comparing CUDA and Julia outputs more strongly
than the current structural `tools/verify_coverage.py` fallback. The paper's
tables and figures should consume both the existing coverage/porting records
and the new numerical-accuracy records described in §11.

---

## 2. Current state — what already exists

### 2.0 The `${STATE}` directory (Codex-specific)

Codex has no session-scoped scratchpad the way Claude Code does, so
this run-book uses a **fixed on-disk state directory**. Every command
in this file that reads or writes `${STATE}/...` refers to that path.

Set it once at the top of your session, then re-export it in any new
shell:

```bash
export STATE="${PORTING_STATE_DIR:-$PWD/.porting-state}"
mkdir -p "$STATE"/{logs,porting_logs}
```

`.porting-state/` is already covered by `.gitignore` (added in this
commit); do not stage anything under it. Contents you write there:

* `${STATE}/coverage.db` — SQLite from `verify_coverage.py`.
* `${STATE}/perf.csv` — CSV from `perf_sweep.py`.
* `${STATE}/logs/*.log` — build/run stdout+stderr.
* `${STATE}/porting_logs/<bench>-julia/` — per-port prompt +
  response + error + manifest (§6).
* `${STATE}/porting_logs/<bench>-julia/context_checkpoint_NN.md` —
  local handoff summary written after each completed port instead of asking
  the user for remote/platform context compression.
* `${STATE}/{cuda,omp,julia}.txt` — worklist snapshots (§3.1).
* `${STATE}/julia_missing.txt`, `${STATE}/batch_A.txt`,
  `${STATE}/batch_B.txt`, `${STATE}/julia_deferred.txt` — Julia worklists.

Two tools currently hard-code an old Claude-session scratchpad path:
`tools/verify_coverage.py` (`DB_PATH`, `log_dir`) and
`tools/perf_sweep.py` (`OUT_CSV`). Before your first run, patch both
to read `${STATE}` from the environment — a two-line change per tool.

### 2.1 What already exists in the repo

Ground truth is the file tree, not this document; verify before trusting.

* Existing Julia ports live in `src/*-julia/`. Treat those directories as
  already done unless the user explicitly asks for rework.
* Other target-language siblings may exist and can be read for reference, but
  they are not part of this Codex pass.
* **Reference sources** for every benchmark live in `src/<bench>-cuda/`,
  `-hip`, `-sycl`, `-omp` — the CUDA one is authoritative for the port.
* **Tooling** (all in `tools/`):
  * `verify_coverage.py` — builds a benchmark for a list of models, runs
    each, and cross-compares stdout with a canonicalizer that (i) strips
    timings, (ii) trusts the bench's own `PASS`/`FAIL`, (iii) falls back to
    structural comparison of numeric summary lines. Writes to a SQLite
    `coverage.db` in the session scratchpad.
  * `perf_sweep.py` — grid over `(bench, model, size)`, runs each cell with
    a per-model timeout, parses the "Average execution time" line, appends
    to `perf.csv`.
  * `perf_sweep_to_tex.py` — turns `perf.csv` into LaTeX rows for
    `2026-PASC-Hecbench-Julia-paper/sections/06_performance_roofline.tex`.
  * `perf_plot.py`, `perf_plot_scaling.py` — matplotlib plots.
  * `hecbench` (CLI) — repo-wide list/build/run wrapper on the CMake tree.
* **Language environments** already set up:
  * Julia: shared project at `src/_julia_env/` with `CUDA.jl` +
    `StaticArrays`. Every `-julia/Makefile` passes
    `--project=/home/imo/HeCBench/src/_julia_env`.
  * Existing non-Julia environments may be present, but they are reference
    material only during this pass.
* **Toolchain paths on this box** (Blackwell RTX 5090):
  * NVIDIA HPC SDK: `/opt/nvidia/hpc_sdk/Linux_x86_64/26.3` → `nvcc`,
    `nvc++`. Use `ARCH=sm_120` for CUDA and `SM=cc120` for OMP nvc++.
  * No local ROCm — HIP builds must be marked `skipped`.
  * No local oneAPI — SYCL builds must be marked `skipped`.
* **Benchmark metadata**: `benchmarks.yaml` at the repo root lists each
  benchmark's category, existing model set, and (for many) the default
  `args` and stdout-parsing regex used by the perf tools.

---

## 3. Scope — Julia coverage for the full CUDA catalogue

**Status: translation tranche complete/triaged.** The previous primary
objective was to create working `src/<bench>-julia/` ports for CUDA
benchmarks without Julia siblings. That objective is finished for the current
pass: the tree has 448 Julia directories, the coverage database records 441
Julia `ok` rows, five existing Julia ports still have `run_fail` rows, and
all 89 CUDA benchmarks still lacking a Julia directory have been recorded as
deferred or out of scope in local state. Future agents should not continue
walking the missing-port list by default.

Only create or repair Julia ports now when the user explicitly asks to reopen
a specific failing or deferred benchmark. Otherwise, use the worklist and
verification commands in this section only as historical context and for
targeted rechecks while the active work proceeds through the numerical
accuracy audit in §11.

Regenerate the exact missing count with the §3.1 commands — it drifts as
ports land. Outside the explicit parallel workflow in §3.2.2 and §4.0, do not
open a second port while one is still failing verification.

### 3.1 The Julia worklist

Regenerate the Julia worklist every time you sit down — some ports may have
landed since the last pass:

```bash
ls src/*-cuda   -d | xargs -n1 basename | sed 's/-cuda$//'   | sort > ${STATE}/cuda.txt
ls src/*-omp    -d | xargs -n1 basename | sed 's/-omp$//'    | sort > ${STATE}/omp.txt
ls src/*-julia  -d 2>/dev/null | xargs -n1 basename | sed 's/-julia$//'  | sort > ${STATE}/julia.txt

# Julia gap
comm -23 ${STATE}/cuda.txt ${STATE}/julia.txt  > ${STATE}/julia_missing.txt

wc -l ${STATE}/julia_missing.txt
```

Persist these lists; the `manifest.json` per port (§6) is what tells you
which entries have already been *attempted* (successfully or not) so you
don't restart from zero on the next session.

### 3.2 Prioritization inside the worklists

Do **not** walk the lists alphabetically. Rank benchmarks so you spend
LLM tokens where they pay off:

#### 3.2.1 Julia prioritization

Apply to `${STATE}/julia_missing.txt`.

1. **Batch A — trivially portable, first**. Benchmarks whose CUDA
   sources are (a) single `.cu` file, (b) < ~400 LoC, (c) no external
   libraries. Grep-friendly test:
   ```bash
   for b in $(cat ${STATE}/julia_missing.txt); do
     files=$(ls src/${b}-cuda/*.cu 2>/dev/null | wc -l)
     loc=$(wc -l src/${b}-cuda/*.cu 2>/dev/null | awk 'END{print $1}')
     ext=$(grep -lE 'boost|gsl|gdal|mpi|nccl|ccl|bz2' src/${b}-cuda/*.cu src/${b}-cuda/*.h 2>/dev/null | wc -l)
     [ "$files" = 1 ] && [ "$loc" -lt 400 ] && [ "$ext" = 0 ] && echo "$b"
   done > ${STATE}/batch_A.txt
   ```
2. **Batch B — has a self-verifier**. From what's left, keep only ports
   whose CUDA source contains `compare_results`, `verify(`, or prints
   `PASS`/`FAIL`. These give you a signal without you having to invent a
   CPU reference:
   ```bash
   for b in $(comm -23 ${STATE}/julia_missing.txt ${STATE}/batch_A.txt); do
     grep -qE 'compare_results|\bverify\b|"PASS"|"FAIL"' src/${b}-cuda/*.{cu,h,cpp} 2>/dev/null && echo "$b"
   done > ${STATE}/batch_B.txt
   ```
3. **Batch C — medium complexity**. The rest of the worklist, minus
   anything on the deferred list (§3.3).
4. **Batch D — deferred**. Do not attempt in this pass; see §3.3.

Work Batch A first, then Batch B, then Batch C. Outside the explicit
sub-agent workflow below, do not open a new port while one is still failing
verification.

#### 3.2.2 Current 20-to-50 benchmark verified short-list

For the current user-requested push, first finish the self-verifying
single-file CUDA benchmarks under 400 LoC. Regenerate this list after every
batch because completed ports disappear from `${STATE}/julia_missing.txt`:

```bash
for b in $(cat ${STATE}/julia_missing.txt); do
  files=$(ls src/${b}-cuda/*.cu 2>/dev/null | wc -l)
  loc=$(wc -l src/${b}-cuda/*.cu 2>/dev/null | awk 'END{print $1}')
  ext=$(grep -lE 'boost|gsl|gdal|mpi|nccl|ccl|bz2' \
        src/${b}-cuda/*.cu src/${b}-cuda/*.h 2>/dev/null | wc -l)
  ver=$(grep -lE 'PASS|FAIL|compare_results|\bverify\b' \
        src/${b}-cuda/*.{cu,h,cpp} 2>/dev/null | wc -l)
  [ "$files" = 1 ] && [ "$loc" -lt 400 ] && \
    [ "$ext" = 0 ] && [ "$ver" -gt 0 ] && printf '%04d %s\n' "$loc" "$b"
done | sort -n > ${STATE}/batch_verified_lt400.txt
wc -l ${STATE}/batch_verified_lt400.txt
```

Treat the regenerated list as the first candidate pool for the current
tranche. Each agent run should complete at least 50 additional Julia ports
when enough feasible CUDA benchmarks remain. If fewer than 50 candidates are
available in this verified-under-400-LoC pool, continue into Batch A, then
Batch B, then Batch C, skipping only benchmarks that meet the deferred rules
in §3.3. If regenerated counts differ from previous runs, trust the file
tree and record the new counts in `${STATE}/logs/worklist_YYYY-MM-DD.log`.

Parallel execution rules for this tranche:

* The coordinator assigns each sub-agent a small disjoint slice, preferably
  3-5 benchmarks at a time. Assign from
  `${STATE}/batch_verified_lt400.txt` first, then refill workers from
  Batch A/B/C worklists as needed to reach the 50-port objective.
* Each sub-agent owns only its assigned `src/<bench>-julia/` directories and
  matching `${STATE}/porting_logs/<bench>-julia/` paths. It must not edit
  unrelated benchmarks, shared tools, existing Julia ports, or non-Julia
  targets unless the coordinator explicitly reassigns ownership.
* Sub-agents may implement and run native `make run` checks independently.
  The coordinator performs or re-runs final
  `tools/verify_coverage.py <bench> --models cuda,julia --only-existing`
  checks before committing.
* GPU verification can be resource-bound. If parallel CUDA/JIT runs become
  unstable, serialize the final verifier commands while keeping code
  implementation parallel.
* If a sub-agent hits a failing port that needs more than two repair
  iterations, it writes an in-flight checkpoint, records the current error
  tag, and returns that benchmark to the coordinator instead of blocking the
  whole tranche.

### 3.3 Deferred / out-of-scope benchmarks

Skip and record in **`${STATE}/julia_deferred.txt`** with a one-word reason
anything that hits:

* **External native libs we don't have on this box**: Boost, GSL, GDAL,
  MPI/NCCL, CCL, BZip2. `README.md` at the repo root lists which
  benchmarks pull each of those in. The corresponding Julia bindings
  either don't exist or aren't worth the yak-shave.
* **cuBLAS / cuFFT / cuRAND / cuSPARSE / cuDNN calls in the CUDA source**
  — CUDA.jl has `CUBLAS`, `CUFFT`, `CURAND`, `CUSPARSE`, `CUDNN`
  submodules that mirror these. **Prefer them** to hand-rolled kernels
  for the Julia port. Only defer if the CUDA source uses a library API
  that CUDA.jl doesn't expose (grep the CUDA.jl source under
  `~/.julia/packages/CUDA/` if in doubt).
* **Multi-file kernel sources with heavy `__device__` inlining
  (>1500 LoC across `.cu` + `.cuh`)** — port cost dwarfs the paper's
  value from that data point. Note them and move on.
* **Benchmarks whose input data isn't checked in and isn't pullable
  through `dvc`** — you can't verify. Skip the Julia port.
* **Work that requires creating or repairing another target language** —
  defer with reason `not-julia-scope`.

The deferred lists *are* data points for the paper: they quantify which
CUDA idioms don't translate cheaply, and belong in the paper's §2
taxonomy.

### 3.4 Extending scope beyond Julia (not this pass)

Only if the user asks you to. Until then, non-Julia targets are frozen.

---

## 4. The Julia porting loop

### 4.0 Parallel coordinator mode for the 50-benchmark push

Use this mode only when the user has explicitly requested sub-agents or
parallel benchmark work. The coordinator remains responsible for global
correctness:

1. Regenerate `${STATE}/batch_verified_lt400.txt` (§3.2.2) plus Batch A/B/C
   worklists.
2. Remove entries that already have `src/<bench>-julia/` or a passing
   manifest in `${STATE}/porting_logs/<bench>-julia/manifest.json`.
3. Spawn workers with disjoint ownership, for example:
   ```
   Worker A owns: mcpr, bscan, zoom
   Worker B owns: channelShuffle, addBiasResidualLayerNorm, bilateral
   Worker C owns: lif, mrc, dp4a
   ```
   Workers must be told they are not alone in the codebase and must not
   revert or overwrite edits outside their assigned paths.
4. Each worker implements only its assigned ports, runs native Julia checks,
   writes manifests/checkpoints, and reports changed files plus verification
   evidence.
5. The coordinator reviews returned changes, reruns or performs final
   cross-model verification, commits passing ports in coherent batches, and
   updates local checkpoints with commit SHAs.
6. Failed or oversized ports are not allowed to stall the tranche. Record
   their taxonomy tag and move them to `${STATE}/julia_deferred.txt` or an
   in-flight checkpoint, then continue assigning remaining feasible entries
   until 50 more ports are complete or the feasible worklist is exhausted.

Even in parallel mode, do not accept a port as complete without native
`PASS` and cross-model `MATCH` evidence unless it is explicitly marked
deferred or in-flight.

Work **one benchmark at a time**. For this pass the only target is
`julia`. In single-agent mode, do not open a second port until the current
Julia port is verified, logged, committed, and followed by a local context
checkpoint. In parallel coordinator mode, this rule applies per worker and
per assigned benchmark directory rather than globally.
Follow the context hygiene section during the loop: store verbose evidence in
local log files and keep the active conversation short enough that no remote
compaction is needed.

### 4.1 Bootstrap the port directory

```
src/<bench>-julia/
  Makefile
  main.jl
```

* **Read the CUDA source first** (`src/<bench>-cuda/*.cu` + any `.h`) —
  that is the semantic spec.
* **Optionally skim existing siblings** (`-omp`, `-serial`, etc.) if they
  already exist; they can be useful references. Do not create or modify
  non-Julia siblings during this pass.
* **Preserve the timed region**. The CUDA source has an explicit
  `chrono::steady_clock` around `cudaDeviceSynchronize` → kernels →
  `cudaDeviceSynchronize`. The Julia port must reproduce the same timed
  region (see §4.5).
* **Preserve the verification**. If the CUDA source runs a CPU reference
  and calls `compare_results<T>` from `include/util.h`, do the same — call
  the CPU reference from the Julia port and emit a single `PASS`/`FAIL` line.
  Do not silently downgrade the check.

### 4.2 Makefile skeleton

The verify + perf tools drive **`make run`** with the CUDA benchmark's
canonical arguments. Use every run line the CUDA Makefile exercises.

```make
JULIA ?= julia
PROJECT ?= --project=/home/imo/HeCBench/src/_julia_env
LAUNCHER ?=

main:
	@true

clean:
	rm -rf __pycache__

run: main
	$(LAUNCHER) $(JULIA) $(PROJECT) main.jl <args>
```

### 4.3 Julia (`CUDA.jl`) gotchas

These aren't optional taste — they're the failure modes previous Julia ports
hit.

* **Julia (`CUDA.jl`) — READ FIRST FOR THIS PASS**
  * All indices in kernels are 1-based on the Julia side but the CUDA
    source is 0-based. Do the conversion once at the array-access site
    with `+ Int32(1)` and comment it, don't sprinkle it.
  * Kernel functions must return `nothing` (write bare `return` at the
    end).
  * Use `Int32` for anything crossing to the device. `Int` on this Linux
    box is 64-bit and blows up in NVRTC / PTX lowering. `blockIdx().x`
    et al. return `Int32` already; keep the arithmetic in `Int32`.
  * `CUDA.@allowscalar d_x[1]` is the only way to read a single device
    value on the host cheaply — use it for `over`/`done`/reduction flags.
  * For a kernel doing an atomic increment, use `CUDA.@atomic` on a
    device array element; for `atomicMax` on an `Int32` counter, prefer
    `CUDA.atomic_max!(pointer(arr, i), Int32(v))`.
  * **Match the timed region exactly.** The CUDA source does
    `cudaDeviceSynchronize(); auto t0 = steady_clock::now(); ...kernel
    launches...; cudaDeviceSynchronize(); auto t1 = ...`. Julian
    equivalent:
    ```julia
    CUDA.synchronize()
    t0 = time_ns()
    @cuda threads=... blocks=... kernel!(...)
    CUDA.synchronize()
    dt_us = (time_ns() - t0) * 1e-3
    @printf("Total kernel execution time : %f (us)\n", dt_us)
    ```
    The printed string must match the perf-sweep regex (see §4.5).
  * Prefer CUDA.jl submodules (`CUBLAS`, `CUFFT`, `CURAND`, `CUSPARSE`,
    `CUDNN`) over hand-rolling a kernel when the CUDA source is calling
    the vendor lib. That's what §3.3 refers to.
  * Struct-of-arrays vs array-of-structs: CUDA structs like
    `struct Node { int starting; int no_of_edges; };` should become two
    parallel `CuArray{Int32}` in Julia (SoA). Do **not** define a
    `struct Node` in Julia and pass a `CuArray{Node}` — CUDA.jl can't
    always lower non-primitive element types cleanly.
  * Kernel launches with `Int` fed as `N` (host `Int64`) will silently
    promote registers to 64-bit and hurt occupancy. Cast: `Int32(N)`.
  * Debugging: `CUDA.@device_code_warntype @cuda ...` shows
    type-instability that would otherwise silently produce slow code.

### 4.4 Iteration protocol (LLM-in-the-loop)

You are expected to iterate — most ports **do not compile on the first
try**. The loop is:

1. Draft the port.
2. `make -C src/<bench>-julia clean && make -C src/<bench>-julia`.
3. If build fails → classify the error (§5), fix, save the prompt and
   diff (§6), goto 2.
4. `make -C src/<bench>-julia run` with the standard input.
5. If run fails or emits `FAIL` → classify, fix, save, goto 2.
6. If run emits `PASS` → cross-check against another target with
   `tools/verify_coverage.py <bench> --models cuda,julia --only-existing`
   (§4.6).
7. If cross-check MISMATCH → classify, fix, save, goto 2.

Stop when 4–6 all pass. Record the iteration count, write the manifest,
commit the port, update the Julia worklist, and write a local context
checkpoint before starting another port in the same worker. In parallel
coordinator mode, the coordinator may batch several passing ports into one
commit if each port has independent verification evidence and a checkpoint.

### 4.5 Where the timing goes

For the perf sweep to pick up your port's number, print **one** line
whose shape is `Average [...] time [...] N.NN (us|ms|s)`. `parse_time_us`
in `perf_sweep.py` walks all matches and keeps the last one, so print it
after the last timed region, not in the middle.

### 4.6 Verification (semantic + numerical accuracy)

Two mandatory levels:

* **Bench-native** — the port itself runs a CPU reference and prints
  `PASS`/`FAIL`. Prefer element-wise `==` for integer outputs; for
  float, use the same tolerance the CUDA sibling uses (grep `compare` in
  `src/<bench>-cuda/`). If none is specified, use `abs(a-b) <=
  1e-3 * max(1, abs(b))` and note the choice in the port's docstring.
* **Cross-model** — run:
  ```
  python3 tools/verify_coverage.py <bench> --models cuda,julia --only-existing
  ```
  You may include additional already-existing models, such as `omp` or
  `serial`, as verification references when useful. Do not create missing
  non-Julia models for that purpose.
  This canonicalizes stdout: drops timing lines, keeps `PASS`/`FAIL`,
  compares numeric-stripped data lines as a multiset. A `MATCH` result
  means the ports agree structurally *and* every one of them said `PASS`.
  Coverage rows land in `${STATE}/coverage.db` — commit them to
  memory of the run, they are what feeds §7's paper tables.

---

## 5. Error taxonomy — record every hit

Every iteration that fails must be classified into exactly one bucket.
Store the record in the session scratchpad (see §6). The taxonomy is
fixed:

| Code | Bucket | Meaning |
|------|--------|---------|
| **B1** | Build: missing toolchain | Julia, CUDA, or package environment missing or wrong version. |
| **B2** | Build: syntax / type mismatch | Julia compile error or host/device type mismatch. |
| **B3** | Build: unresolved symbol / API drift | CUDA.jl function/module renamed, moved, or missing. |
| **B4** | Build: kernel compile failure | CUDA.jl/NVRTC/PTX failure inside a device kernel body. |
| **R1** | Runtime: crash / abort | Segfault, panic, `illegal memory access`, `CUDA_ERROR_*`. |
| **R2** | Runtime: silent wrong shape | Ran to completion, but printed lines differ in count/labels from CUDA. |
| **R3** | Runtime: hang / timeout | Killed by the `perf_sweep.py` per-cell timeout. |
| **N1** | Numerical: FAIL emitted | The port's own verifier said `FAIL`. |
| **N2** | Numerical: cross-model mismatch | `PASS` locally but disagrees with CUDA under `verify_coverage.py`. |
| **N3** | Numerical: tolerated drift | Elements differ but within the documented tolerance. Not a failure — recorded because it's part of the port's story. |
| **S1** | Semantic: wrong algorithm | The port implements a related but non-equivalent computation (e.g. dropped an inner term of the update rule). |
| **S2** | Semantic: wrong timed region | Compiles + verifies but the timed region is not the CUDA-equivalent one (missing sync, timer around wrong block). |
| **P1** | Portability compromise | Deliberate documented Julia deviation. Not a failure — recorded so the paper can name it. |
| **T1** | Tooling: verifier bug | The failure was in `verify_coverage.py`/perf tools, not the port. Fix the tool, re-run. |

When you fix an error, tag the commit / prompt log with the code so §7's
taxonomy table can be built by a simple `grep` later.

---

## 6. What to record — prompts, tokens, iterations, timings

For every Julia port, create one directory in the session
scratchpad:

```
${STATE}/porting_logs/<bench>-julia/
    prompt_01.md        # verbatim prompt sent to the LLM
    response_01.md      # verbatim reply (code diff or full file)
    error_01.md         # build/run/verify output that motivated the next iter
    tag_01              # taxonomy code, e.g. "B2"
    prompt_02.md
    ...
    token_usage.jsonl   # per-iteration translation token accounting
    translation_attempts.jsonl  # candidate, compiler feedback, repair, outcome
    numerical_evidence.jsonl    # how accuracy was checked against references
    manifest.json       # summary — see below
    context_checkpoint_01.md
```

`manifest.json` shape (write it *once* at the end of the port; update
it if you re-open the port later):

```json
{
  "bench": "bfs",
  "target": "julia",
  "driver": "codex",
  "driver_version": "codex-cli-<version>",
  "iterations": 4,
  "tokens_in": 12480,
  "tokens_out": 3910,
  "tokens_cached": 8200,
  "tokens_total": 16390,
  "token_accounting": {
    "status": "exact",
    "method": "provider_usage",
    "source": "token_usage.jsonl",
    "scope": "translation including repair iterations",
    "limitations": []
  },
  "wall_time_s": 812,
  "error_tags": ["B2", "B4", "N2"],
  "final_status": "pass",
  "notes": "Any tolerance, vendor-library mapping, or portability caveat."
}
```

After the port is committed, write
`${STATE}/porting_logs/<bench>-julia/context_checkpoint_NN.md`. This is
the local replacement for remote/platform context compression. It should be
short but complete enough for the next Codex agent to continue without the
full conversation:

* completed benchmark and target;
* commit SHA and commit subject;
* files added or changed;
* implementation notes and any semantic deviations;
* build/run/verify commands and final results;
* manifest path and error tags;
* worklist updates;
* unrelated dirty worktree files that were intentionally left alone;
* next workflow note if the user has changed the run-book expectations.

Do not ask the user to compress context remotely, and do not wait for a
remote compact task before starting the next port once the local checkpoint is
written.
For long or troublesome ports, also write an in-flight checkpoint before the
conversation becomes large; include the active failure and the next command to
run so the task can resume from disk in a fresh session.

### 6.1 Mandatory per-benchmark translation token accounting

Record token consumption for **every benchmark translation**, including
source analysis, initial implementation, and all build/run/verification repair
iterations. Save the record before moving to another benchmark, even if the
attempt fails or is deferred. Record later numerical corrections separately
under `${STATE}/numeric_accuracy/fixes/<bench>/`; preserve the original
translation cost and driver provenance.

Append one JSON object per iteration or attributable model call to
`token_usage.jsonl`. Include `iteration`, `phase`, `model`, `tokens_in`,
`tokens_out`, `tokens_cached`, `status`, `method`, `source`, and
`limitations`. Use benchmark-specific usage records or captured session-usage
deltas; never assign a whole shared session's total to each benchmark. Include
attributable worker usage when workers are used, count each model call once,
and document any coordinator/shared usage that cannot be attributed.

Use the best available evidence in this order:

1. **Exact usage:** Read provider/API usage or available CLI/session usage
   records and sum the calls attributable to this benchmark. Map
   `input_tokens` to `tokens_in`, `output_tokens` to `tokens_out`, and
   `input_tokens_details.cached_tokens` to `tokens_cached`. For usage records
   using `prompt_tokens` and `completion_tokens`, map them to input and output.
   Set `status: "exact"` and `method: "provider_usage"` only when the
   benchmark's full translation usage is captured.
2. **Tokenizer estimate:** If exact usage is unavailable, tokenize the saved
   prompts and responses with a tokenizer matching the model when available.
   Count inputs and outputs separately. Save the tokenizer name/version and
   file list; use `status: "estimated"`, `method: "tokenizer"`.
3. **Character estimate:** If no suitable tokenizer is available, estimate
   input and output separately as `ceil(character_count / 4)` over the saved
   prompt and response text. Count decoded text characters, not raw bytes;
   `wc -c` alone is not a token count. Use `status: "estimated"`,
   `method: "characters_div_4"`, and record character totals and file lists.
4. **Unknown:** If no usable usage records or prompt/response logs exist,
   store `null` for unavailable counts, `status: "unknown"`,
   `method: "unavailable"`, and explain what evidence is missing.

Sum iteration records into the manifest's `tokens_in` and `tokens_out`.
Set `tokens_total = tokens_in + tokens_out` only when both are available;
otherwise set it to `null`. Cached input is a subset of input tokens, so do
not add `tokens_cached` again. Set cached tokens to `null` when unavailable;
never infer them from text length. If only part of the translation is
captured, set manifest accounting status to `partial`, identify the captured
scope, and describe the omitted work. If exact and estimated records are
combined, label the aggregate `estimated` and document the mixed methods.

**Never use zero as a placeholder for unavailable consumption.** A zero is
valid only when evidence establishes zero consumption for that field. Save
actual prompts/responses as work proceeds so estimates remain possible;
never reconstruct missing conversations and present them as original logs.
Text-based estimates measure the saved text and may omit repeated context,
system instructions, tool exchanges, and reasoning tokens; list these limits
and do not present the estimate as exact or billed usage.

The manifest must include `token_accounting` with `status`, `method`, `source`,
`scope`, and `limitations`, as illustrated above. Include token totals,
accounting status, and record paths in each local context checkpoint. For
cost-study tables, distinguish exact, estimated, partial, and unknown records;
do not treat missing values as zero.

For historical backfills, preserve valid recorded counts and provenance.
Replace unavailable zero placeholders with estimates only when saved evidence
supports them; otherwise mark them unknown with `null` counts. Record the
backfill date, method, sources, and limitations in `token_accounting`. Never
attribute later repair usage to the original translation.

Always set `driver: "codex"` and populate `driver_version` (the CLI
version string, or the model id if you're calling the API directly).
The paper's per-driver split in §7 depends on this field being set on
every port you produce.

Runtime performance goes into `perf.csv` (via `perf_sweep.py`), not the
per-port log. That keeps the port log about *cost of producing the
port* and the CSV about *cost of executing it*.

---

### 6.2 Compiler-in-the-loop attempts for every benchmark and Julia version

The paper must analyze how many translation attempts and compiler-feedback
repairs were required for each benchmark in CUDA.jl, KA, and JACC. Keep the
legacy `iterations` field, but state its definition and link it to attempt IDs;
model turns, candidate implementations, compiler invocations, and benchmark
execution trials are different counts.

- A **translation attempt** is a submitted candidate source revision evaluated
  for compilation/execution/correctness, including the initial candidate and
  each repaired candidate. Assign a monotonically increasing `attempt_id` per
  benchmark/version/phase. Compiler rejection still counts as an attempt.
- Record every actual compiler/build/JIT check as a separate invocation linked
  to that candidate. Repeating a command on unchanged source is another
  invocation, not another candidate. A repair prompted by compiler diagnostics
  is a compiler-feedback repair round; distinguish numerical/runtime repairs.
- A Julia `make main` no-op does not prove compilation. Capture the first
  actual host/device JIT compilation or kernel-launch check and its result.
  Record when compilation and execution occur in one command. Do not infer
  internal compiler invocation counts that the tooling does not expose.
- The five performance trials, profiler replays, package installation, and
  unchanged-source revalidation do not create new translation candidates.
  Track environment/setup work and native CUDA reference builds separately.
- Distinguish original translation, later numerical corrections, optimization,
  and migration phases. Preserve failed/abandoned attempts and their costs.
  Do not add later repair attempts to the original historical count silently.

Append `translation_attempts.jsonl` to each variant's operational log directory.
Each record must include benchmark/model/phase, attempt and parent attempt IDs,
source revision or hashes, timestamp, linked model iteration/prompt/response,
compiler/JIT command and version/flags, invocation IDs and exit codes,
diagnostic/log paths, primary failure taxonomy code when applicable,
feedback-driven edit or next action, token-usage record IDs, native run result,
numerical-evidence IDs, and outcome. Capture build success separately from
execution and numerical success; an unsupported attempt is not a verified port.

Add manifest summaries: `translation_attempts`, `compiler_check_invocations`,
`compiler_feedback_repairs`, `runtime_repairs`, `numerical_repairs`,
`attempt_to_first_compile_success`, `attempt_to_first_verified_success`, and
`attempt_accounting` (definition, phase, source paths, completeness/limitations).
Record unknown historical values as `null` with partial/unknown provenance;
never reconstruct attempt counts from tokens, the five timing trials, or
commit counts alone. Native CUDA has no agent translation-attempt count in
this study: mark it not applicable, while retaining its reference build logs.

### 6.3 Evidence explaining how numerical accuracy was confirmed

For every benchmark/version, save `numerical_evidence.jsonl` linked to the
translation attempt, source hash, configuration, and execution-trial IDs. Record:

- Reference identity/revision and exact command for native CUDA and CUDA.jl,
  plus their agreement or disagreement and any conditional build requirements.
- Input file/hash or generation procedure/seed/RNG, dimensions, precision,
  initialization, outputs compared, and evidence of identical input values.
  Matching seed numbers alone do not establish identical RNG output.
- Verification method: native CPU/reference check, numerical stdout regex,
  element-wise array comparison, or artifact comparison; name the checker and
  version and explicitly state what outputs it does and does not inspect.
- The exact acceptance rule, rtol/atol, reference side, integer/exact criteria,
  reduction/error metric, checked element/value count, maximum absolute and
  relative errors and any RMS error, with nonfinite/shape checks as applicable.
  Unknown or inapplicable metrics are null with a reason. Nonfinite results
  cannot pass merely because a floating-point comparison silently ignores NaN.
- Per-reference and per-trial verdicts, raw output/artifact/log paths or hashes,
  deviations, and remaining limitations. Check all five trials even though
  trial 1's execution time is excluded from performance aggregates.

For native checks, preserve the actual verifier algorithm and tolerance and
report its strength rather than assuming the printed PASS checks all outputs.
A compiler success or structural stdout match does not confirm numerical
accuracy. State whether accuracy was established by the native check, summary
agreement, or direct result comparison; retain weaker evidence labels and
conditional results in the paper. Historical evidence gaps stay explicit.

---

## 7. Feeding the paper

The active manuscript is `2026-PASC-Hecbench-Julia-paper/`, with separate
section files as specified in paper_writing.md. The commands below describe
legacy data generation; adapt their output paths and measurement protocol
before using them in the PASC study.

**Per-driver cost-study separation.** Every port produced by Codex is
recorded with `"driver": "codex"` in its `manifest.json` (§6). The
paper's cost-study tables (iterations, tokens, error taxonomy hits)
have *separate columns* for Claude and Codex — do not merge them, and
**do not overwrite a Claude-authored port's manifest** with Codex-run
metrics. If you re-verify a Claude-authored port under Codex (fine,
encouraged as a checkpoint), append a `"reverified_by": "codex"` field
plus a fresh timestamp but leave the original iteration/token counts
alone.

* **§4 performance comparison** — build with:
  ```
  python3 tools/perf_sweep.py \
    --benches accuracy adam adjacent atan2 aidw \
    --models cuda julia \
    --out ${STATE}/perf.csv
  python3 tools/perf_sweep_to_tex.py
  # → ${STATE}/perf_rows.tex, paste into 4.perfomrance_comparison.tex
  ```
  Extend `BENCHMARKS`/`MODELS` in `perf_sweep.py` when you add a new row.
  For the Julia-coverage pass specifically, you do **not** need to add
  every one of the 505 newly ported benchmarks to the perf sweep — that
  would balloon the table beyond what the paper can render. Pick a
  representative subset (roughly 20–30, covering algorithms/graph/
  linear-algebra/image/crypto categories per `benchmarks.yaml`) once
  the coverage sweep is done.
* **§2 taxonomy of porting errors** — one row per `error_tags` bucket
  from §5 for Julia ports, cell = count of ports that hit that bucket
  at least once. No script exists yet; a small Python script over
  `${STATE}/porting_logs/*-julia/manifest.json` produces it.
  Write that script the first time the paper asks for the table, name it
  `tools/taxonomy_to_tex.py`, and commit it.
* **§1/§2 Julia coverage table** — new for this pass. Two columns
  (Julia-Claude, Julia-Codex),
  rows are: "successfully ported" (target `PASS` + cross-model
  `MATCH`), "in flight" (built, not verified), and "deferred /
  out-of-scope" broken down by reason from §3.3 (`external-lib`,
  `vendor-lib-unmapped`, `oversized`, `no-input`, `not-julia-scope`).
  Source of truth is `${STATE}/coverage.db` (`verify_status` table)
  plus the deferred lists plus the `driver` field in each
  `manifest.json`. Build the table with a small Python script — commit
  it as `tools/coverage_to_tex.py` and let it take
  `--target julia` and `--driver claude|codex|all` flags.
* **§2 taxonomy per driver** — the error-taxonomy table
  (§5 buckets for Julia) is also **duplicated per driver** in the paper.
  When you extend `tools/taxonomy_to_tex.py`, take the same `--driver`
  flag as `coverage_to_tex.py`.

The **diagram** (system diagram in §1) is not yet in the tex tree. If the
user asks you to draft it, sketch it as a Mermaid or TikZ block that
mirrors §4's loop: `CUDA source → LLM iteration loop → port + logs →
verify_coverage / perf_sweep → paper tables`.

---

## 8. Commit convention

One commit per Julia port. Message form (from `git log`):

```
<bench>-julia: Julia port (CUDA.jl)

Generated-By: OpenAI Codex
```

Existing commits on this branch may carry a
`Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>`
trailer instead — that is how the paper's per-driver split in §7
resolves ties when a `manifest.json` is missing. Use the `Generated-By`
trailer above for every Codex-authored commit so the two are easy to
partition:

```bash
git log --grep='Generated-By: OpenAI Codex' --oneline   # Codex-authored
git log --grep='Co-Authored-By: Claude'   --oneline     # Claude-authored
```

Examples that already exist:

```
softmax-julia: Julia port (CUDA.jl)
```

Rules:

* Stage only the files under `src/<bench>-julia/` plus any tooling
  change the Julia port needed. Do **not** stage `build/`, `.stamps/`,
  `target/`, or `__pycache__/` — `.gitignore` already excludes them but
  a stray `git add -A` will pull them in.
* Do not commit `${STATE}/**` — the porting logs and coverage db live
  under `.porting-state/` (gitignored) by design.
* Never `--amend` a published commit. If a port has to be revised, make
  a new commit `<bench>-julia: fix <what>` on top. Never touch a
  commit produced under the other driver — the paper's cost columns
  depend on those commits being stable, and rewriting one loses the
  `Co-Authored-By` / `Generated-By` provenance.
* After committing, write
  `${STATE}/porting_logs/<bench>-julia/context_checkpoint_NN.md` before
  continuing to the next port.

---

## 9. Worked examples

### 9a. Julia — porting one missing benchmark end-to-end

Take the first entry of `${STATE}/batch_A.txt` — say it is `xsbench`
(illustrative). `src/xsbench-cuda/` exists, `src/xsbench-julia/` doesn't.

1. `ls src/xsbench-cuda/` — inspect the source layout, note `.cu`,
   headers, any `data/xsbench/*` input file, and the `run:` target in
   its Makefile (that's the canonical input args). Confirm no external
   libs are pulled in (grep for
   `boost|gsl|gdal|mpi|nccl|ccl|bz2|cublas|cufft`).
   If cuBLAS/cuFFT/cuRAND/cuSPARSE/cuDNN show up, plan to use the
   matching CUDA.jl submodule (§3.3) instead of hand-rolling.
2. Read the CUDA source fully. Identify: (a) the kernel(s), (b) the
   timed region, (c) the CPU verifier + tolerance.
3. Optionally read existing siblings for verifier and host-loop context. Do
   not create or modify non-Julia siblings.
4. Create `src/xsbench-julia/{main.jl, Makefile}` from the templates in
   §4.2. Write kernels as `@cuda` functions returning `nothing`; keep
   the CUDA kernel structure line-for-line so the two are easy to diff.
   Watch the Julia gotchas in §4.3 (1-based indexing, `Int32`, SoA over
   AoS, matching timed region).
5. `make -C src/xsbench-julia run` — expect a first-try failure. Common
   ones: `Int64`/`Int32` mismatch across the kernel boundary (`B2`),
   `CUDA.jl` API name (`B3`), off-by-one in the 1-based/0-based
   translation (`N1` or `N2`).
6. Save each `(prompt, response, error, tag)` under
   `${STATE}/porting_logs/xsbench-julia/` per §6.
7. When `make run` prints `PASS`, run:
   ```
   python3 tools/verify_coverage.py xsbench \
     --models cuda,julia --only-existing
   ```
   Expect `MATCH`. If `MISMATCH` and the mismatch is float noise, either
   (i) tighten `compare_outputs`'s canonicalizer (that's a `T1`), or
   (ii) accept it as `N3` and document in the port's docstring.
8. Write `manifest.json`, then commit:
   ```
   xsbench-julia: Julia port (CUDA.jl)
   ```
9. Update the worklist:
   ```
   grep -v '^xsbench$' ${STATE}/julia_missing.txt \
     > ${STATE}/julia_missing.txt.tmp && \
     mv ${STATE}/julia_missing.txt{.tmp,}
   ```
10. Write a local checkpoint:
    ```
    ${STATE}/porting_logs/xsbench-julia/context_checkpoint_01.md
    ```
    Include commit SHA, files changed, final verification output, manifest
    path, worklist updates, and remaining unrelated worktree changes.
11. Start the next entry of Batch A only after that local checkpoint exists.
    Do not ask the user to run a remote/platform context compression task.

Mass-cadence expectation: a Batch A benchmark should take under an hour
of wall-clock and under ~30k tokens per port when it's genuinely
trivial. If any single port blows past 2× either budget, mark it deferred
(§3.3, tag `oversized`) and move on — the paper's story is the
distribution of cost, not chasing every long-tail case.

---

## 10. Fast reference — the commands you'll use most

```bash
# 1. Regenerate the Julia worklist (do this every session)
ls src/*-cuda   -d | xargs -n1 basename | sed 's/-cuda$//'   | sort > ${STATE}/cuda.txt
ls src/*-omp    -d | xargs -n1 basename | sed 's/-omp$//'    | sort > ${STATE}/omp.txt
ls src/*-julia  -d 2>/dev/null | xargs -n1 basename | sed 's/-julia$//'  | sort > ${STATE}/julia.txt
comm -23 ${STATE}/cuda.txt ${STATE}/julia.txt  > ${STATE}/julia_missing.txt
wc -l ${STATE}/julia_missing.txt

# 2. Build Julia Batch A (single-file, <400 LoC, no external libs)
for b in $(cat ${STATE}/julia_missing.txt); do
  files=$(ls src/${b}-cuda/*.cu 2>/dev/null | wc -l)
  loc=$(wc -l src/${b}-cuda/*.cu 2>/dev/null | awk 'END{print $1}')
  ext=$(grep -lE 'boost|gsl|gdal|mpi|nccl|ccl|bz2' src/${b}-cuda/*.{cu,h} 2>/dev/null | wc -l)
  [ "$files" = 1 ] && [ "$loc" -lt 400 ] && [ "$ext" = 0 ] && echo "$b"
done > ${STATE}/batch_A.txt
wc -l ${STATE}/batch_A.txt

# 2b. Current parallel tranche: verified single-file, <400 LoC candidates
for b in $(cat ${STATE}/julia_missing.txt); do
  files=$(ls src/${b}-cuda/*.cu 2>/dev/null | wc -l)
  loc=$(wc -l src/${b}-cuda/*.cu 2>/dev/null | awk 'END{print $1}')
  ext=$(grep -lE 'boost|gsl|gdal|mpi|nccl|ccl|bz2' src/${b}-cuda/*.{cu,h} 2>/dev/null | wc -l)
  ver=$(grep -lE 'PASS|FAIL|compare_results|\bverify\b' src/${b}-cuda/*.{cu,h,cpp} 2>/dev/null | wc -l)
  [ "$files" = 1 ] && [ "$loc" -lt 400 ] && [ "$ext" = 0 ] && [ "$ver" -gt 0 ] && printf '%04d %s\n' "$loc" "$b"
done | sort -n > ${STATE}/batch_verified_lt400.txt
wc -l ${STATE}/batch_verified_lt400.txt

# 3. Build + run one Julia port
make -C src/<bench>-julia clean && make -C src/<bench>-julia
make -C src/<bench>-julia run

# 4. Cross-verify one Julia port against CUDA
python3 tools/verify_coverage.py <bench> \
  --models cuda,julia --only-existing

# 5. Bulk-verify the whole existing Julia set
for b in $(ls src/*-julia -d | xargs -n1 basename | sed 's/-julia$//'); do
  python3 tools/verify_coverage.py "$b" \
    --models cuda,julia --only-existing 2>&1 \
    | tee -a ${STATE}/logs/julia_sweep.log
done

# 6. Perf sweep for a hand-picked subset
python3 tools/perf_sweep.py --benches <bench1> <bench2> ... \
  --models cuda julia

# 7. Regenerate the LaTeX perf rows for the paper
python3 tools/perf_sweep_to_tex.py

# 8. Query Julia verification history from a prior session
sqlite3 ${STATE}/coverage.db \
  "SELECT bench, model, status, detail FROM verify_status
   WHERE model = 'julia' ORDER BY bench, model;"
```

---

## 11. Active work: numerical correctness fixes

The Julia porting push has now completed the broad coverage tranche:
`${STATE}/coverage.db` currently records **441 Julia ports with verified
`ok` results** under `tools/verify_coverage.py`. That result means the
ports build/run and the existing cross-model verifier accepts the CUDA and
Julia outputs. It does **not** guarantee strict numerical equivalence for
benchmarks that lack a native `PASS`/`FAIL` verifier, because
`verify_coverage.py` falls back to structural stdout comparison after
normalizing numeric literals.

The numerical-audit tooling is now in place; the next pass is therefore a
**numerical correctness correction pass**. Treat CUDA as the reference output.
Do not replace the existing coverage verifier; use the stronger numerical
layer alongside it and fix or document every failed/gap row.

To-do for the next agent:

1. **Classify every Julia port by verification strength.**
   Split `src/*-julia/` into: (a) ports with native `PASS`/`FAIL`, (b)
   ports that only print numeric stdout summaries, (c) ports that write
   output files/images/arrays, and (d) ports that cannot be numerically
   checked until missing inputs or external dependencies are available.
   Persist the classification under `${STATE}/numeric_accuracy/` and
   summarize counts in `${STATE}/logs/numeric_accuracy_classification.log`.

2. **Add a dedicated numeric verifier.**
   Create `tools/verify_numeric.py` to build/run CUDA and Julia, capture
   stdout/stderr, compare non-timing numeric values with tolerances, compare
   configured output artifacts, and write results to
   `${STATE}/numeric_accuracy.db`. The tool should accept at least:
   ```
   python3 tools/verify_numeric.py <bench> \
     --models cuda,julia --only-existing
   ```
   and a bulk mode or shell-friendly loop should work across all existing
   Julia ports.

3. **Extend benchmark metadata for numeric checks.**
   Add `numeric_check:` entries to `benchmarks.yaml` as needed. Supported
   metadata should cover stdout regexes, relative/absolute tolerances, and
   output files. Example shape:
   ```yaml
   minimod:
     numeric_check:
       stdout_regex:
         - 'Checksum: min_u,  max_u = ([0-9eE+.-]+), ([0-9eE+.-]+)'
       rtol: 1.0e-5
       atol: 1.0e-7

   dxtc2:
     numeric_check:
       files:
         - path: data/lena_std.dds
           kind: binary
           tolerance:
             rms: 1.0e-2
   ```
   Start with the recently added/known weak-verification ports
   (`bm3d`, `dxtc2`, `memtest`, `merkle`, `minimod`, `miniWeather`) and
   then sweep the rest.

4. **Add deterministic comparable summaries when needed.**
   If CUDA and Julia do not currently print comparable numeric results,
   add a small deterministic checksum or summary only when it can be added
   consistently to both sides or is already present in the CUDA output.
   Do not silently invent a Julia-only correctness criterion and call it
   equivalent to CUDA. Any deliberate tolerance or summary choice must be
   documented in the port log and tagged `N3` or `P1` as appropriate.

5. **Run artifact-producing benchmarks in isolated directories.**
   Some benchmarks overwrite tracked or shared output files. The numeric
   verifier must run CUDA and Julia in temporary per-model work directories
   under `/tmp` or `${STATE}/numeric_accuracy/runs/`, then compare outputs
   there. Do not allow a numeric sweep to dirty tracked source directories
   with generated images, binary outputs, `.dat` files, or solver dumps.

The first numerical sweep has been run for all 441 coverage-`ok` Julia
ports. The baseline is recorded in `JULIA_CORRECTNESS_CHECK.md`,
`${STATE}/numeric_accuracy.db`, and
`${STATE}/logs/numeric_accuracy_summary_2026-09-24.md`:

```text
pass|native_pass_trusted|251
pass|numeric_stdout|37
mismatch|numeric_stdout|76
mismatch|native_verdict|1
unverified|no_numeric_data|56
not_run|run_status|19
not_run|timeout|1
```

Current correction target:

* Investigate and fix or reclassify the `77` mismatch rows.
* Add benchmark-specific numeric evidence for the `56`
  `unverified|no_numeric_data` rows.
* Repair or retest the `20` `not_run` rows.
* Keep `JULIA_CORRECTNESS_CHECK.md` synchronized with each completed batch.

### 11.1 Plan for correcting failed numerical testing

1. **Freeze the baseline.**
   Keep `JULIA_CORRECTNESS_CHECK.md` as the top-level status. Treat
   `${STATE}/numeric_accuracy.db` as the source of truth for current
   numerical status. Export and maintain three working lists under
   `${STATE}/numeric_accuracy/reports/`: mismatches, unverified/no-data, and
   not-run.

2. **Separate false positives from real failures.**
   Inspect `mismatch|numeric_stdout` rows first. Many may be timing,
   performance numbers, nondeterministic summaries, IDs, or metadata that
   `tools/verify_numeric.py` is incorrectly treating as numerical results.
   Fix verifier filtering before touching ports. Reclassify false mismatches
   as `unverified|no_numeric_data` when no real result exists, or as
   `pass|numeric_stdout` when only harmless formatting/noise caused the
   mismatch.

3. **Add benchmark-specific `numeric_check` metadata.**
   For benchmarks with real numeric summaries, add regex/tolerance rules in
   `benchmarks.yaml`. Prioritize recently added or known weak ports:
   `bm3d`, `dxtc2`, `memtest`, `merkle`, `minimod`, and `miniWeather`, then
   work through mismatch rows alphabetically or by category.

4. **Handle artifact-producing benchmarks safely.**
   Extend `tools/verify_numeric.py` to run CUDA and Julia in isolated
   temporary directories. Add artifact comparison support: binary exact
   compare, numeric text/table compare, and image RMS/MAE comparison. Use this
   for ports that write `.png`, `.dds`, `.dat`, `.nrrd`, `.pgm`, or similar
   artifacts.

5. **Repair true Julia numerical bugs.**
   For each confirmed mismatch, read the CUDA source as the spec, compare the
   Julia implementation, and fix indexing, type width, launch geometry,
   memory layout, tolerance, RNG, or algorithmic differences. Then run native
   `make run`, `tools/verify_coverage.py`, and `tools/verify_numeric.py`.
   Write a local checkpoint and commit one benchmark fix at a time.

6. **Fix not-run rows.**
   Split not-run rows by cause: CUDA build failure, CUDA run failure, Julia
   run failure, and timeout. Do not mark numerical failure until the CUDA and
   Julia pair runs. Repair Julia run failures first; record CUDA-reference
   failures separately.

7. **Resolve unverified rows.**
   For `unverified|no_numeric_data`, add comparable checksums when CUDA
   already exposes equivalent output, add artifact metadata when files are
   generated, or leave the row explicitly documented as unverified. Do not
   claim numerical correctness without comparable evidence.

8. **Use multi-agent batches.**
   Assign workers disjoint chunks by category: verifier false-positive
   cleanup, artifact-comparison support, not-run repair, confirmed Julia bug
   fixes, and metadata/checksum additions. Each worker writes local
   checkpoints under `${STATE}/numeric_accuracy/fixes/<bench>/`. Each
   committed source fix should be one benchmark per commit and must not stage
   `.porting-state/`.

9. **Regenerate the final report.**
   After each correction batch, rerun:
   ```
   python3 tools/verify_numeric.py \
     --bench-list ${STATE}/numeric_accuracy/julia_verified_ok.txt
   python3 tools/verify_numeric.py --summary
   ```
   Update `JULIA_CORRECTNESS_CHECK.md` and
   `${STATE}/logs/numeric_accuracy_summary_YYYY-MM-DD.md`. The target is to
   maximize `pass|native_pass_trusted`, `pass|numeric_stdout`, and
   artifact-pass rows, minimize true mismatches, and explicitly document
   unavoidable unverified cases.

Record final numerical-correction status in `${STATE}/numeric_accuracy.db` and
write a concise checkpoint under
`${STATE}/logs/numeric_accuracy_summary_YYYY-MM-DD.md` with counts for:
native verifier trusted, numeric stdout matched, artifact matched,
numeric mismatch, missing inputs, external dependency, and not yet
classified.

---

## 12. What NOT to do

* Do not open ports for target languages other than Julia during this pass.
* Do not port benchmarks the user hasn't asked for — if the CUDA source
  hits any §3.3 deferral condition, add it to
  `${STATE}/julia_deferred.txt` with a one-word reason and move on.
* Do not generate Serial ports as a prerequisite for Julia. Existing
  non-Julia siblings may be read or included in verification only if they are
  already present.
* Do not "improve" the CUDA reference — even if you spot a bug, treat the
  CUDA port as the spec. File the observation in the port's docstring
  and move on.
* Do not silently downgrade a `FAIL` to a `PASS` by loosening a
  tolerance. Any tolerance change goes in the port's docstring and is
  tagged `N3` in the log.
* Do not commit `.porting-state/`, the `build/` tree, the `.stamps/`
  directories, `target/`, or `__pycache__/`.
* Do not run destructive git operations (`reset --hard`,
  `push --force`, `checkout .`) without confirming with the user
  first — even in Codex's `--dangerously-bypass-approvals-and-sandbox`
  mode. Auto-approve does not remove the need to think before
  destroying local work.
* Do not touch existing `src/*-julia/` ports that were authored by another
  driver (see §0). Editing them would corrupt the paper's per-driver cost
  split. Add new Julia ports on top; leave existing ones alone unless the user
  explicitly re-scopes them to you.
* Do not rewrite git history on this branch. The 186 commits already
  on `monil/refactor_coverage` carry the `Co-Authored-By: Claude`
  provenance that §7's per-driver tables key on — rewriting them would
  break the paper's data.
* In single-agent mode, do not open a second port while a previous port is
  still failing verification — batching drops signal about which iteration
  cost went where and pollutes the `manifest.json` records. In explicit
  parallel coordinator mode (§4.0), multiple sub-agents may work
  concurrently only on disjoint assigned benchmark directories.
* Do not continue to the next Julia port in the same worker after a
  successful commit until you have written the local context checkpoint under
  `${STATE}/porting_logs/<bench>-julia/`. In parallel mode, the coordinator
  may integrate other workers' completed ports while one worker writes its
  next checkpoint.
* Do not ask the user to run a remote/platform context compression task as
  part of the normal Julia-port cadence; local checkpoints are the handoff
  mechanism.


---

## 13. KA and JACC versions of the 300 passing CUDA.jl benchmarks

### 13.1 Scope, names, and frozen baseline

Create two additional implementations per selected benchmark, for a target of
600 new variants:

| Implementation | Source directory | Model identifier |
|---|---|---|
| CUDA reference | `src/<bench>-cuda/` | `cuda` |
| CUDA.jl reference | `src/<bench>-julia/<bench>-julia-cuda.jl/` | `julia` |
| KernelAbstractions.jl | `src/<bench>-julia/<bench>-julia-KA/` | `julia-ka` |
| JACC.jl | `src/<bench>-julia/<bench>-julia-JACC/` | `julia-jacc` |

The directory spelling and case above are intentional; `cuda.jl` is part of
that directory name, not a single source file. Each child contains its own
`main.jl`, Makefile, and required local assets. Example:

```text
src/<bench>-julia/
  <bench>-julia-cuda.jl/
    main.jl
    Makefile
  <bench>-julia-KA/
    main.jl
    Makefile
  <bench>-julia-JACC/
    main.jl
    Makefile
```

Keep model identifiers stable for historical databases (`julia`, `julia-ka`,
`julia-jacc`); do not derive model identity from capitalization or leaf suffixes.
The CUDA C++ source remains `src/<bench>-cuda/`.

**Transition:** Existing CUDA.jl ports currently use flat files such as
`src/<bench>-julia/main.jl`. Before starting that benchmark's translation,
record the actual source paths, revisions, and file hashes. Adapt discovery,
build/run commands, project paths, relative data paths, ontology source paths,
and verification/performance tooling to the nested layout before moving files.
Migrate one benchmark at a time, preserving CUDA.jl content, history, manifests,
tokens, baseline IDs, and numerical conditions. Record old-to-new paths in
`${STATE}/portable_julia/source_layout.json` and the combined checkpoint.
Keep an intentional compatibility wrapper at the parent if existing tooling
needs it; count only child implementations, never wrappers, as ports. During
the transition, resolve the nested CUDA.jl child if present, otherwise the
legacy flat implementation; the parent directory alone does not prove a
CUDA.jl implementation exists. Do not copy or regenerate the CUDA.jl reference
and attribute migration as a new translation. This instruction change defines
the target layout; the source migration is a later implementation phase.

Use the official APIs and record package versions/revisions:
- KA: https://github.com/JuliaGPU/KernelAbstractions.jl
- JACC: https://github.com/JuliaGPU/JACC.jl

Before implementation, freeze the 300 existing `status='pass'` rows from
`${STATE}/numeric_accuracy.db` whose CUDA.jl source directories exist. Write
`${STATE}/portable_julia/baseline_300.json` and `baseline_300.txt`; include
benchmark names, verification methods/details, timestamps, source revisions
or hashes, input arguments, tolerances, and required build conditions.
If the live count changes, retain the original snapshot and document the
change; do not silently substitute benchmarks. The snapshot is a selection
baseline, not proof of raw-array equivalence for every benchmark.

At this update the 300 comprise 259 `native_pass_trusted` and 41
`numeric_stdout` results. Native checks vary in strength, and stdout matching
covers only exposed summaries. `atan2` passes summary comparison only with
CUDA `--fmad=false`; its default-build integer RMSE summary mismatch remains
unresolved. Preserve this condition in every derived comparison. Do not
claim all 300 have identical-input, element-wise CUDA equivalence.

Retain the original CUDA and CUDA.jl source and provenance. Read both as
references; implement the new variant in its own directory using KA or JACC
for the translated computation. Keep vendor-library dependencies explicit.
A wrapper that invokes the existing CUDA.jl benchmark is not a new port.
Use separate dependency environments/preferences for KA and JACC, select the
CUDA backend explicitly, and record the device actually used. Cross-vendor
portability requires separate hardware validation.

### 13.1.1 Translation order: complete both variants per benchmark

Work one benchmark at a time, implementing **KA first, then JACC for that
same benchmark**, before starting the next benchmark. Keep the algorithm,
reference behavior, and validation configuration consistent across the pair.

1. Read the CUDA and CUDA.jl implementations. Establish the shared inputs,
   numerical criteria, reference build conditions, and timing boundaries.
2. Implement `src/<bench>-julia/<bench>-julia-KA/`. Verify against both references, collect
   the five execution trials required by §13.3, and save its separate tokens,
   iteration logs, correctness/performance results, manifest, commit, and
   context checkpoint.
3. Implement `src/<bench>-julia/<bench>-julia-JACC/` using the same established configuration.
   Repeat dual-reference verification and five-trial timing, and save its own
   tokens, iterations, results, manifest, commit, and context checkpoint.
4. Update pair progress, feature limitations, and the paper observation/claim
   records required by paper_writing.md. Regenerate affected paper evidence or
   checkpoint the exact pending analysis. Save the combined §14 checkpoint.
   Select the next benchmark only after both variants have recorded outcomes
   and checkpoints and the paper state is captured.

If a required feature is unsupported in one framework, document that variant
under §13.4 and continue with the other framework for the same benchmark.
For another unresolved blocker, save an explicit failure/deferred status,
evidence, token usage, and an in-flight checkpoint; a recorded blocked outcome
allows the other variant to proceed without falsely claiming completion.

Begin with a small representative set spanning element-wise operations,
reductions, shared memory, atomics, and multiple kernels. Complete each
benchmark's KA/JACC pair in this order, use the pilot to establish the shared
workflow, then continue through the frozen 300-benchmark list. This order
also applies within each worker's assignment if parallel work is explicitly
authorized; keep timed GPU trials serialized.

### 13.2 Verification against CUDA and CUDA.jl

For each benchmark, build the unchanged CUDA reference, run the existing
CUDA.jl reference, and run both available new variants. Use identical problem
sizes, repeat counts, precision, input data/seeds, initialization, and numerical
criteria. When seeded generators differ, use shared deterministic inputs or
record the limitation and the actual verification strength. Preserve equivalent
kernel timing boundaries and synchronization. Record every compiler flag,
package version, backend, GPU architecture, command, and reference revision.

Record independent comparisons for `julia-ka` versus `cuda`, `julia-ka` versus
`julia`, `julia-jacc` versus `cuda`, and `julia-jacc` versus `julia`, plus the
CUDA-versus-CUDA.jl reference agreement. A variant passes only when its runs
succeed and both reference comparisons pass under the recorded conditions.
If the references disagree, record `reference_disagreement` and investigate;
do not silently select whichever reference matches the variant. Conditional
reference results must retain their conditions and remain visible in reports.

Preserve native verifiers and compare actual numerical results, arrays, or
artifacts where available, using documented absolute/relative tolerances.
Record whether evidence is `native_pass_trusted`, `numeric_stdout`, or artifact/
element-wise comparison. Structural stdout matching alone is insufficient.
If only native verdicts or summaries are available, retain that verification
strength explicitly. No missing evidence, failed check, or unsupported feature
may be relabeled as a pass. Isolate artifact-producing runs per model/trial.

Current tools require extension before use for these model identifiers:
`verify_numeric.py` currently compares only CUDA and CUDA.jl and its database
is keyed by benchmark alone. Extend discovery, execution, comparison, and
storage for variants; do not claim existing two-model commands verify KA/JACC.
Use `${STATE}/portable_julia/verification.db` with records keyed by benchmark,
variant, reference, configuration, and trial. Preserve the original
`${STATE}/numeric_accuracy.db` baseline. Link raw evidence from every result.

### 13.3 Five runs and execution time

**Required protocol: `five_trials_last_four_v1`.** For each benchmark/model/
configuration, collect one complete set of five execution trials. Record the
protocol ID, run count (5), excluded trial (1), and retained trials ([2, 3, 4, 5])
with each result. Exclude only trial 1's time and report the arithmetic mean
of trials 2–5. This is the sole required execution-timing protocol; do not
collect an additional measurement series. Only complete, matched five-run
sets qualify for the paper's runtime/performance analysis.

For **every selected benchmark and available model** (`cuda`, `julia`,
`julia-ka`, `julia-jacc`), perform five numbered execution trials with the
same configuration while verifying correctness. Run sequentially on the same
GPU without competing benchmark jobs. Build once, reuse the initialized
process for the five trials where possible, and reset inputs, mutable state,
and RNG between trials. Retain the CUDA benchmark's internal repetition count;
these are five outer trials, not five unrelated internal iterations.

- Trial 1: warm-up; save output, verdict, and elapsed time; exclude its time
  from performance aggregates.
- Trials 2, 3, 4, and 5: save output, verdict, and elapsed time; report
  `mean_time_us = (trial_2_time_us + trial_3_time_us + trial_4_time_us + trial_5_time_us) / 4`.
- Correctness must pass on all five trials, including the warm-up. Retain
  each measured time, its unit, timing region, and aggregation formula.
- Synchronize the device immediately before and after each timed region.
  Match the CUDA timed region and normalize units to microseconds. Record
  process wall time separately from kernel/benchmark execution time.
- Handle Julia JIT and library initialization consistently. If separate
  processes are necessary, warm/compile within each process before measuring
  its execution region and record this deviation. Discarding only the first
  process does not remove JIT costs from later fresh processes.
- If a required trial fails, times out, or has no trustworthy execution time,
  retain its status and use `null` for the aggregate; do not substitute zero,
  silently drop failed trials, or report a speedup for a failing variant.

Store `${STATE}/portable_julia/timings.csv` with benchmark, model, backend,
configuration ID, trial index, warm-up flag, execution time in microseconds,
correctness status, timing-region description, and raw-log path. Write a
summary CSV with all four retained trial times and their mean for each model.
Record speedups only for comparable timing regions/configurations with passing
correctness, keeping raw execution time available. Apply this protocol to all
300 selected benchmarks, including their two references, rather than a paper
performance subset. The current performance tools must be adapted to implement
this protocol before their output can be used as evidence of compliance.

The paper's performance comparison must explicitly include **native CUDA
(C++), CUDA.jl, KA, and JACC** for each benchmark/configuration. Export one
comparison row with all four models' measured times (trials 2, 3, 4, 5, and mean), their
correctness/evidence/conditional statuses, and their shared configuration ID.
For each Julia implementation define speedup versus native CUDA as
`t_native_cuda / t_implementation` (greater than one is faster). Also report
comparisons among CUDA.jl, KA, and JACC to answer abstraction-selection questions.
Keep kernel-region execution time distinct from compile/JIT, startup, transfers,
and end-to-end time; compare a given metric only over equivalent regions.

Use correctness-qualified matched configurations for aggregate comparisons;
report the common benchmark set, denominator, missing/unsupported cases, and
any pairwise subset. Keep conditional build configurations separate. Do not
silently drop failures or compare old CUDA timings with new variants measured
on a different GPU, input, compiler configuration, or timing region.

### 13.4 Translation tokens and unsupported features

Use separate records for each new variant:
`${STATE}/porting_logs/<bench>-julia-ka/` and
`${STATE}/porting_logs/<bench>-julia-jacc/`. Apply §6.1 to both, including
analysis, implementation, failed attempts, repairs, and unsupported/deferred
attempts. Set `target` to `julia-ka` or `julia-jacc`, preserve benchmark/model
identity, and record the driver and model version. Store per-iteration
`token_usage.jsonl`, a manifest with input/output/cached/total tokens and
accounting status/method/sources/limitations, and a context checkpoint before
starting the next variant. Shared work must be attributed once or recorded as
unallocated shared usage. Preserve original CUDA.jl translation consumption.
Use exact usage where available, labeled estimates otherwise, and `null` for
unknown counts. Token estimates based on summaries remain `partial`.

If KA or JACC cannot express a required feature, create an explicit record in
`${STATE}/portable_julia/unsupported_features.jsonl` and the variant manifest.
Include benchmark, variant, backend, package version/revision, required feature,
CUDA source location, reason, documentation/source evidence, attempted API,
minimal failing example or error/log path, workaround assessment, date, and
translation tokens spent. Examples to investigate include warp intrinsics,
specialized hardware instructions, atomics, synchronization, shared memory,
and vendor-library interoperability. Inspect the installed API before claiming
a feature is unsupported; support may depend on version/backend.

Distinguish `unsupported_feature`, `missing_dependency`, `missing_input`,
`build_fail`, `run_fail`, `numeric_mismatch`, `unverified`, and
`reference_disagreement`. A tooling/environment failure is not evidence that
the framework lacks a feature. Document any workaround, its numerical and
performance consequences, and whether it adds backend-specific code. Label
such implementations as hybrid/backend-dependent rather than fully portable.
Do not silently replace the computation with a different algorithm or a
CUDA.jl fallback and count it as a pure KA/JACC success.

### 13.5 Completion and reporting

The active paper is `2026-PASC-Hecbench-Julia-paper/main.tex`, using ACM
`sigconf` with separate section files. Follow paper_writing.md §11 for the
user's clean, readable, compact chart style, the common CUDA/CUDA.jl/KA/JACC
code example immediately after the introduction, and the combined methodology
diagram linking compiler feedback, numerical checks, ontology, evidence,
performance characteristics, and paper checkpoints. Preserve draft/result
status and do not draw fabricated performance charts.

The manuscript is organized around stable RQ1–RQ5 and E1–E5 defined in
[paper_writing.md](paper_writing.md) §3: feasibility/numerical correctness,
compiler-in-the-loop effort/failures, four-model performance, ontology-linked
mechanisms, and abstraction selection. Tag observations, claims, result
snapshots, and checkpoints with `rq_ids` and `experiment_ids`. Update the RQ
answer matrix in the paper's research/protocol.md after each batch, recording
available evidence, missing analysis, answer status, and manuscript location.
An RQ is answered by traceable results and their limitations, not by completing
a port count. Decision synthesis alone is not an independent experiment.

For each variant, commit its source with a `Generated-By: OpenAI Codex`
trailer, update its manifest and checkpoint with the commit, reference checks,
five trial results/mean time for the four-model comparison, compiler-in-the-
loop attempt counts/history (§6.2), numerical-evidence method/results (§6.3),
token accounting, and feature limitations. Link these records into the paper's
per-benchmark observations and generated analysis.
Keep `.porting-state/` untracked. Shared tooling changes should be distinct
from benchmark commits. Parallel workers require explicit user authorization;
if authorized, assign disjoint variant directories and serialize timed runs.

Update `JULIA_CORRECTNESS_CHECK.md` after each batch with separate CUDA.jl,
KA, and JACC counts and verification strengths; timings, token-accounting
coverage, and unsupported-feature counts/benchmarks must be traceable to local
records. Report conditional passes separately. A 300-entry CUDA.jl baseline
does not imply either new framework already has 300 passing ports.


---

## 14. Combined translation and paper checkpointing

Paper progress is a mandatory part of checkpointing. Follow
[paper_writing.md](paper_writing.md) for the full record fields. Keep existing
per-variant checkpoints and add a durable combined checkpoint under:

```text
2026-PASC-Hecbench-Julia-paper/research/checkpoints/
  checkpoint_<UTC timestamp>.md
  LATEST.json
```

Write after each completed variant, each completed benchmark pair and its
paper update, each meaningful paper-only batch, and before ending a session
or a context handoff. Save in-flight state for interrupted or failing work.
Write the checkpoint completely first, then atomically replace LATEST.json
with its ID, repository-relative path, timestamp, and status. Update the
pointer only after referenced records are written. Retain older checkpoints.

Every combined checkpoint must contain:

- Branch/HEAD and relevant commits for both the translation repository and
  the separate nested paper repository, changed/uncommitted files, and unrelated
  dirty work to preserve; distinguish source, paper, and documentation edits.
- Canonical instruction paths and hashes/revisions; actual STATE path,
  baseline snapshot, input/configuration IDs, and source-layout map/status.
- Active benchmark and variant, paths, KA/JACC pair progress, outcomes,
  unsupported features, and exact next implementation command/action.
- Verification against both references, conditions/tolerances, all five trial
  times (first excluded from the mean), result/log/DB paths, failures, and
  running-job/session IDs. Record absent measurements as unknown.
- Per-variant tokens, iterations, candidate attempt IDs/counts, compiler/JIT
  check and feedback-repair counts, first compile/verified success, accounting
  completeness and log paths; distinguish translation/correction phases.
- Numerical-evidence method, identical-input evidence, acceptance rule,
  tolerances/error metrics, checked counts, both reference verdicts, and the
  matched four-model performance table/data snapshot.
- Shared effort and migration effort, preserving original translation costs.
- Paper section files changed, RQ1–RQ5 answer status, E1–E5 evidence and
  pending analysis, observation/claim IDs with evidence and confidence,
  ontology record/version, tables/figures and their data/code
  snapshot, remaining analysis, citations, and manuscript build status.
- Separate concrete next actions for translation and writing, plus their
  dependencies. Record which remains mandatory before the next benchmark.

Keep raw runs/databases/transcripts under STATE and store concise portable
summaries in paper/research. Never stage `.porting-state/`. Durable checkpoints
should reference source snapshots and record unavailable external artifacts
rather than pretending git includes local databases. Resume from the same
variant/paper phase without relying on chat history or remote compaction.

### 14.1 Active cohort restart tools

Generate a combined checkpoint with `python3 tools/checkpoint_portable_study.py`. Refresh portable paper results with `python3 2026-PASC-Hecbench-Julia-paper/research/analysis/export_verified_study.py --source-root .`. Read local `portable_julia/batch_100.json` for current scope, ownership, and rejection/refill history. The 100-pair active goal is incomplete until100 legitimate benchmark configurations meet bothreference/five-trial requirements and the actual-results paper is complete. Semantic reference repairs preserve the original source and record separate correction-phase costs. Named subkernel/precision coverage remains explicit; never claim entireCLI coverage from a single configuration.

Cumulative Codexsession usage counters may be available but shared concurrent translation, infrastructure, reference repair, and writing cannot be attributed exactly to individual variants without explicit boundaries. Save sharedusage separately; do not replace pervariant unknowns with unallocatedsession totals.


### Active study resource-policy checkpoint (2026-10-16)

For kernels without shared data, explicitly set JACC `launch_spec(shmem_size=0, ...)`. Installed JACC1.4.0 uses automatic maximum dynamic shared allocation when omitted; a controlled triad experiment measured49152B/block and a shared-memory occupancy limit2, versus0B/limit32 after this change, with unchanged geometry/registers and allfive fulloutput checks passing. Kernels requiring shared arrays must request their actual required bytes, never blindly zero them. Treat this as a measured configuration-specific resource-policy finding, not universal framework overhead.

Keep original verified default-resource source/results as an ablation. A changed launch-resource policy is a separately named `performance_tuning` configuration, with tuning attempts/tokens separate from original translation. Reuse unchanged validated reference/KA bundles under exact source, command, input and protocol checks. Any nonsemantic dependency exception requires explicit old/new SHA256 and a reviewed reason; the current exception only adds Int64 artifact typing after timing, leaving Float32/64/Int32 paths unchanged. Rawfive samples and allcorrectness verdicts remain retained. The paper's `research/data/configuration_dispositions.json` selects primary configurations without counting old/new resource-policy versions as independent benchmarks.


## Completed 100-pair study (2026-10-17 UTC)

The requested cohort is complete: **100 benchmarks, 200 separately committed KA/JACC versions, and 107 primary matched configurations**. The five-trial protocol excludes trial 1 and averages trials 2–5 for native CUDA, CUDA.jl, KA, and JACC. All recorded numerical comparisons passed; final numerical, effort, and source-proof audits are in `2026-PASC-Hecbench-Julia-paper/research/review/`. Seventeen additional resource-policy configurations are retained as ablations.

Resume by reading `AGENT_INSTRUCTIONS_CODEX_Julia.md`, `JULIA_CORRECTNESS_CHECK.md`, and `paper_writing.md`, then following `2026-PASC-Hecbench-Julia-paper/research/checkpoints/LATEST.json`. A checkpoint marked `complete_100_pairs_manuscript_review_ready` supersedes historical active-work notes. Do not restart the completed cohort or launch new measurements automatically.

The finished technical manuscript and build instructions are in `2026-PASC-Hecbench-Julia-paper/README.md`; the PDF is `build/main.pdf` inside that repository. Author/publication metadata and an archival artifact release remain submission tasks. The study covers named configurations on one V100, not all CLI modes or all 300 historical ports. Ninety-nine pairs contain translated kernels; MRG32k3a is explicitly a cuRAND interoperability case. Numerical evidence distinguishes complete arrays, defined struct payload, canonicalized multisets, and complex property-score checks.

Per-version compiler checks, source candidates, known repairs, correctness, timing, feature outcomes, and ontology records are preserved. Token records are partial estimates/proxies; exact consumed or billed tokens per version are unavailable and must not be inferred from source size or shared session counters. Missing repair counts remain unknown. Raw numerical artifacts and detailed local logs remain in `.porting-state`; portable results, provenance, audit summaries, and analysis scripts remain in the local ignored research folder; publication files are tracked in the paper repository.

### Revised acceptance contracts

If an auxiliary numerical checker is revised after observing a native result, preserve the original criterion, rejected captures, source/input hashes and derivation. Freeze a separately named secondary contract before a fresh five-trial series for all four implementations. Set `verification_contract_revised: true` and `exclude_untouched_native_aggregates: true`; a checker-only revision does not set `native_reference_repaired`. Record the post-observation change as shared verification infrastructure work, separate from authored GPU candidates and compiler feedback. Report the revised-contract stratum separately in the registry, checkpoint and paper. Never bypass exact input/acceptance contract reuse checks to import an older series.

### Actual compiler and library alignment

Bind the compiler, headers and resolved runtime libraries for each measured configuration, including vendor bridges called by Julia. Record actual loaded library paths, versions and file hashes; a declared project, link flag or RUNPATH alone does not establish the loaded version. Numerical agreement does not resolve a library-version performance confound. Preserve a mismatched series as historical evidence, exclude it from the controlled performance comparison, freeze an explicitly versioned replacement configuration and measure the affected versions again. Strictly unchanged versions may be reused only through the documented, hash-checked cross-configuration mechanism. Report alternate native compiler/toolkit strata separately from the pinned environment and include them as covariates in the paper. Count loader/environment repairs separately from authored GPU candidates, device compiler feedback and numerical repairs.

### Reproducible source archives

Before staging an original-source or candidate-history directory, inspect its file inventory. Retain sources, licenses, independent CPU fixtures, generators, patches and essential runtime inputs. Keep compiled executables, object files, shared libraries, raw GPU trial captures and profiler caches in local ignored evidence; filter binary magic bytes rather than relying on filename extensions. Record their hashes and rebuild commands. Preserve existing local artifacts when removing accidental Git tracking. Paper publication commits exclude the local research tree.
