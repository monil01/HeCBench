# PASC27 Julia translation paper — writing instructions

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


### Honest page budgets and repair subtypes

Before a paper checkpoint, flush body floats before the bibliography and inspect the rendered PDF. PASC's ten-page body limit includes every figure, table and appendix, including any float otherwise deferred beyond the references. Record body and reference page counts separately. Keep ACM fonts and margins; transfer supporting analyses to the clearly labeled working technical companion. This companion has no implied conference supplement or archival status.

In failure tables and case studies distinguish native compile-only compatibility changes, synchronization/allocation corrections and mathematical/scientific repairs. A build rejection is not a numerical defect. Exact-zero overload disambiguation can be semantically neutral yet remains a disclosed changed-native intervention in the conservative conditional stratum; do not pool it silently with untouched CUDA or claim all conditional cases have faulty scientific formulas.



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

Writing priority: reorganize the scientific narrative now, retaining valid pilot results as an explicitly labeled interim cohort. Add family-by-model coverage and failure panels, implementation-specific ontology comparison examples, and a domain-facing contribution statement. Update every claimed count from audited data; write pending300 analyses as questions/protocol rather than completed results.

---


## 1. Entry point, destination, and authority

The canonical restart entry is
[AGENT_INSTRUCTIONS_CODEX_Julia.md](AGENT_INSTRUCTIONS_CODEX_Julia.md).
Read it together with [JULIA_CORRECTNESS_CHECK.md](JULIA_CORRECTNESS_CHECK.md)
and this file, then follow the
[latest combined checkpoint](2026-PASC-Hecbench-Julia-paper/research/checkpoints/LATEST.json).
This is the paper-writing companion, including when the user refers to
"paper writing.md". The exact filename is `paper_writing.md`.

Use **`2026-PASC-Hecbench-Julia-paper/`** as the manuscript destination.
Ignore its existing draft's narrative and results when designing the new
paper; do not inherit claims without evidence. Preserve existing files until
an explicit, documented replacement is implemented. Keep each section in a
separate `.tex` file. The old `Hecbench-agent-paper/` is not this paper's
active destination. The active ACM manuscript is now `2026-PASC-Hecbench-Julia-paper/main.tex`.
Its separate section files, illustrative code comparison, and methodology
diagram are present. The generated census and newest checkpoint identify current KA/JACC results; older pilot-only passages are explicitly labeled interim evidence.

Authoritative division of responsibility:

- Agent instructions §§13–14: source layout, translation order, operational
  records, verification/timing/token rules, and combined checkpoint contract.
- JULIA_CORRECTNESS_CHECK.md: numerical status and strength, conditional
  results, coverage gaps, and current verification/reporting instructions.
- This file: research questions, manuscript structure, observations/claims,
  ontology analysis, research limitations, and writing/checkpoint cadence.
- Raw records and source revisions: evidence for all empirical statements.
  Resolve disagreements with documents by inspecting evidence and recording
  the correction; never silently change a result to match a draft sentence.

## 2. Venue and section files

Target PASC27. The official guidelines checked on 2026-10-16 specify ACM
`sigconf`, a first-stage limit of 10 pages including figures/tables/appendices
but excluding references, and full-paper submission on 2026-12-12 (AoE).
Verify current rules before formatting/submission:
https://pasc-conference.org/pasc27/submission/guidelines-for-papers/

Use a root `main.tex` with `\input{sections/...}` and one section per file:

```text
2026-PASC-Hecbench-Julia-paper/
  main.tex
  references.bib
  sections/
    00_abstract.tex
    01_introduction.tex
    02_background_related_work.tex
    03_agent_translation_workflow.tex
    04_benchmarks_ontology_methodology.tex
    05_translation_cost_failures.tex
    06_performance_roofline.tex
    07_abstraction_selection.tex
    08_limitations_reproducibility.tex
    09_conclusion.tex
  generated/
  figures/
  research/
    protocol.md
    claims.jsonl
    observations/<bench>.md
    snapshots/
    analysis/
    checkpoints/
      checkpoint_<UTC timestamp>.md
      LATEST.json
```

The active root `main.tex` uses `\documentclass[sigconf]{acmart}` and inputs
the separate section files above. Existing IEEEtran drafts are preserved as
legacy material. Author/publication metadata and empirical results remain
placeholders; inspect the latest checkpoint for actual build status. Keep generated table
fragments in `generated/` and their generating code under `research/analysis/`
or shared `tools/`; figures must link to the same analysis snapshot.

## 3. Research questions, experiments, and answers

Organize the paper around the following stable IDs, **RQ1–RQ5**. Introduce them
in the introduction, map the experimental methods to them, and organize results
and discussion around explicit answers. A result section must state the question,
experiment/design, evidence, answer, and limitation. Do not organize results as
a chronological translation diary or claim an answer before its evidence exists.

### RQ1 — Translation feasibility and numerical correctness

**Question:** To what extent can the agent produce faithful CUDA.jl, KA, and
JACC implementations, and how is numerical correctness confirmed?

**E1 — Translation and correctness evaluation:** For the frozen benchmark set,
record each variant's outcome, required/unsupported features, reference build
conditions, and evidence from native checks, numerical summaries, or direct
array/artifact comparisons. Verify new variants against both native CUDA and
CUDA.jl using matched input values, precision, and documented acceptance rules.
Report the weaker evidence levels and conditional cases explicitly.

**Answer artifacts:** Per-benchmark correctness/evidence table; success,
conditional-success, failure, unverified, and unsupported counts by abstraction
and ontology feature; representative numerical-failure/repair cases. Include
all selected benchmarks in outcome denominators. Preserve the selection bias
and historical CUDA.jl evidence limitations.

### RQ2 — Translation effort and compiler-in-the-loop failures

**Question:** How many compiler-in-the-loop attempts and tokens are required
per benchmark/version, and which failure mechanisms account for that effort?

**E2 — Translation-process analysis:** Record candidate attempts, actual
compiler/build/JIT checks, compiler-feedback repairs, runtime/numerical repairs,
attempt to first compile and first verified success, tokens, elapsed effort,
and human interventions. Join attempt histories with the fixed error taxonomy
and ontology features. Include failed and unsupported attempts and their costs.

**Answer artifacts:** Per-benchmark attempt/token table; first-attempt compile
and verified-success rates; effort distributions and taxonomy breakdowns by
abstraction and workload features. Keep exact/estimated/partial/unknown data
separate. Production KA-then-JACC costs are incremental and order-dependent;
historical CUDA.jl costs cannot automatically serve as a controlled comparison.

### RQ3 — Execution performance across the four implementations

**Question:** How does execution performance differ between native CUDA,
CUDA.jl, KA, and JACC across benchmarks and problem sizes?

**E3 — Matched runtime experiments:** Run each benchmark/model/configuration
five times under `five_trials_last_four_v1`; check correctness on all five,
exclude trial 1's time, and average trials 2–5. Match hardware, inputs,
precision, arithmetic/build conditions, and timed regions. Keep compile/JIT
and startup costs separate from execution time. Retain individual timings.

**Answer artifacts:** Four-model timing table, native-relative speedups,
Julia-to-Julia comparisons, size-scaling plots, and matched-set summaries with
sample counts and missing/unsupported/conditional outcomes. Claims of small
performance differences require adequate variability evidence.

### RQ4 — Ontology-linked explanations of performance

**Question:** Which workload characteristics and measured implementation
mechanisms explain where abstractions differ in performance?

**E4 — Ontology-stratified profiling and mechanism experiments:** Freeze/review
motif, memory-access, synchronization, precision, and required-feature labels
before examining rankings. Compare E3 runtime ratios within these categories
and profile representative implementations for work/traffic, roofline position,
launches, synchronization, locality, and resource use. Investigate paired
within-benchmark differences and test proposed mechanisms with controlled
changes where feasible, preserving correctness and recording extra effort.

**Answer artifacts:** Motif/access/synchronization-stratified runtime plots,
paired roofline panels where appropriate, feature/performance associations,
and source-to-execution case studies. Use suitable bandwidth/latency/operation
metrics for non-floating-point workloads. Distinguish inferred labels,
measured counters, correlations, and experimentally supported mechanisms;
account for hardware/size/precision and related benchmark families. Runtime-
derived roofline efficiency is not independent evidence of a runtime cause.

### RQ5 — Choosing an abstraction for a use case

**Question:** Given a workload's required features and its correctness,
performance, and translation-effort constraints, which abstraction should a
user target, and where is the evidence insufficient to recommend one?

**E5 — Decision synthesis and validation:** Combine E1–E4 into explicit
constraint-based choices and cost/performance trade-offs. State application
requirements and any decision thresholds before evaluating the choices.
Report ties, trade-offs, unsupported cases, and uncertainty; avoid inventing
one universal winner. Test any proposed predictive selection rule on reserved
benchmark families or applications, with selection/training separated from
evaluation. Without such validation, label the matrix as descriptive guidance
for the observed population rather than an independently validated predictor.

**Answer artifacts:** Evidence-backed use-case/abstraction decision matrix,
trade-off plots, examples/counterexamples, and validation results where available.
E5 is synthesis unless a separate validation experiment is actually executed;
do not imply another experiment occurred merely because the table was written.

### 3.1 Manuscript organization and evidence traceability

- Introduction: motivation, contributions, and exact RQ1–RQ5 wording.
- Background: Julia abstractions and relevant work needed to interpret RQs.
- Agent workflow and methodology: agent definition, frozen population, ontology,
  experimental designs E1–E5, controls, and evidence/accounting definitions.
- `sections/05_translation_cost_failures.tex`: explicit RQ1 correctness/coverage
  and RQ2 effort/failure results in named subsections.
- `sections/06_performance_roofline.tex`: explicit RQ3 four-model performance
  and RQ4 ontology/mechanism results in named subsections.
- `sections/07_abstraction_selection.tex`: RQ5 synthesis and validation status.
- Limitations: scope, selection, numerical strength, historical missing costs,
  translation order, timing uncertainty, and hardware-generalization limits.
- Conclusion: a concise evidence-backed answer to each RQ, with no new claims.

Use the separate TeX files specified in §2. Map each experiment to its source
population/configurations, independent comparison unit, metrics, controls,
evidence snapshot, analysis command, output table/figure, and relevant RQ IDs
in research/protocol.md. Observations, claims, snapshots, and checkpoints must
include `rq_ids` and `experiment_ids`; each empirical manuscript claim must link
to its analysis snapshot and question. Multiple RQs may share one result bundle;
this does not multiply the number of experiments or independent samples.

Maintain an RQ answer matrix in research/protocol.md with columns for RQ,
experiment, available evidence, missing analysis, manuscript location, answer
status, and limitations. Status is `pending`, `partial`, or `answered` according
to evidence, independently of translation progress. Keep unresolved questions
visible; do not force a definite recommendation from missing or weak data.

### 3.2 Agent description and writing cadence

Describe the actual agent: model/version, reasoning configuration when exposed,
instruction and prompt revisions, source inputs and their order, tools,
package/backend versions, attempt/repair policy, stop conditions, time/token
budgets, and human interventions. Define what constitutes an iteration and a
failure event. Describe observed behavior rather than an idealized workflow.
Record operational changes as protocol versions, retaining which runs used each.

Write methods and background while translation proceeds. Write results only
from evidence, and revise the abstract/conclusion after the main results are
stable. Leave explicit TODOs instead of invented measurements or citations.

## 4. Source layout and experimental population

Use one parent per Julia benchmark:

```text
src/<bench>-julia/
  <bench>-julia-cuda.jl/  # model: julia
  <bench>-julia-KA/       # model: julia-ka
  <bench>-julia-JACC/     # model: julia-jacc
```

The original CUDA reference stays in `src/<bench>-cuda/`. Child names are
case-sensitive directories, each with its own implementation and Makefile.
Legacy flat CUDA.jl sources may still exist until their per-benchmark migration.
Use the actual path and source-layout map recorded by agent instructions §13.1.
Moving files is migration effort, not a new CUDA.jl translation or numerical
result. Preserve original source hashes, author attribution, and token records.

Freeze the 300-entry passing CUDA.jl baseline before new translation. As of
this guide, 259 records trust native checks and 41 compare numerical summaries;
`atan2` requires CUDA --fmad=false for its passing summary comparison.
Do not describe this as 300 unconditional element-wise equivalence results.
Selection from successful CUDA.jl ports limits conclusions about the full
CUDA catalogue. Preserve all failed/unsupported attempts within the selected
population and report their denominators.

## 5. Work cycle: one benchmark, KA then JACC, then paper

Start with representative benchmarks covering maps, reductions, shared memory,
atomics, and multiple kernels. For each benchmark:

1. Read CUDA, CUDA.jl, and `ontology/benchmarks/<bench>.yaml`. Record source
   hashes, ontology revision, reference conditions, inputs, and protocol ID.
2. Implement KA, verify against both references, collect the prescribed
   five-trial timings, and record tokens/iterations and every failed attempt.
   Save its manifest and per-variant checkpoint, including current paper state.
3. Implement JACC for the same benchmark/configuration and record the same
   evidence independently. If unsupported, save API/version/failure evidence,
   tokens spent, and a precise status; continue with the other variant.
4. Create/update `research/observations/<bench>.md`, link the result bundle to
   ontology properties, update candidate claims, and update correctness and
   pair progress. Assign RQ1–RQ5 and E1–E5 IDs to the evidence and update the
   RQ answer matrix. No result may be omitted because it contradicts a hypothesis.
5. Regenerate affected tables/figures when evidence is sufficient; otherwise
   record the exact pending analysis and missing inputs. Update relevant paper
   sections with supported observations or explicit TODOs.
6. Save the combined checkpoint and advance LATEST.json before the next
   benchmark. Both variant outcomes and paper observation state are required.

After a small batch (default: five benchmark pairs, or the end of a session),
refresh the summary figures/tables, review claims and contradictions, and revise
results/discussion. A paper-only session must also checkpoint its progress.
Translation, analysis, and writing can be interleaved; GPU timings remain
serialized. Parallel workers require the user's explicit authorization.

## 6. Observations, claims, and failure taxonomy

Each benchmark observation must record:

- Benchmark/variant/configuration IDs, `rq_ids` and `experiment_ids`, source
  and ontology revisions, protocol version, run/log/DB paths, and timestamp.
- Native versus summary versus array/artifact correctness, both reference
  comparisons, conditions, raw five-trial times and retained mean, tokens,
  iteration count, and the quality/completeness of token evidence.
- Actual observation; proposed explanation; alternative explanations; any
  counterexample; next experiment; and the relevant paper section/question.
- Failed attempts and unsupported features, workaround/backend dependencies,
  algorithm or timing-region changes, and remaining numerical limitations.

Use stable claim IDs in `research/claims.jsonl`. A record includes claim text,
`rq_ids`, `experiment_ids`, observation/evidence IDs, analysis snapshot, supporting and
contradicting evidence, limitations, status (`hypothesis`, `supported`,
`contradicted`, or `unresolved`), manuscript locations, and next action.
A performance difference is an observation; attributing it to the abstraction
requires checking compiler, configuration, library, algorithm, and timing
confounds. Phrase causal claims only as strongly as experiments support.

Retain the agent instructions' fixed B/R/N/S/P/T error taxonomy. Classify each
failed iteration once in its primary bucket, link it to source/feature evidence,
and record phase, symptom, root cause, repair, and outcome. Keep unsupported
feature status and ontology relationships as additional fields, not silently
renamed error buckets. Distinguish failure-event counts, benchmarks affected,
failed attempts, and final failures. Include failed/deferred effort in costs.

### 6.1 Per-benchmark compiler-in-the-loop analysis

Follow the common definitions in agent instructions §6.2. Report separately
for each benchmark and Julia version: candidate translation attempts (initial
candidate included), actual compiler/build/JIT check invocations, compiler-
feedback repair rounds, runtime and numerical repair counts, and the attempt
reaching first compilation and first verified success. Keep model iterations
and timing trials separate. Link every compiler diagnostic to the source
candidate, subsequent edit, error-taxonomy bucket, tokens, and final outcome.
A no-op Makefile target is not evidence of successful device JIT compilation.

Generate per-benchmark tables and per-abstraction distributions, first-attempt
compile/verified-success rates, and attempt/cost breakdowns by ontology features
and failure taxonomy. Include failed/unsupported attempts and explicit unknown
historical values; disclose denominators and accounting completeness. Preserve
original CUDA.jl translation histories and separate later corrections. Native
CUDA is the performance/reference baseline, not another agent-translated target.
Do not infer attempts from commit counts or the five performance trials.

### 6.2 Explain numerical confirmation in the paper

For each benchmark/version, link the compiler-attempt history to the numerical
evidence required by agent instructions §6.3 and JULIA_CORRECTNESS_CHECK.md.
Record shared input values or hashes/generation/RNG details, sizes, precision,
both reference identities and commands, the native-verifier algorithm or exact
outputs/arrays/artifacts compared, checker version, acceptance rule and
rtol/atol, checked value/element counts, observed error metrics, nonfinite/shape
handling, and per-reference/per-trial results. Keep raw logs and configuration
conditions traceable from the paper's snapshots.

Write the methodology explaining how accuracy was confirmed and generate a
per-benchmark evidence table. Separate trusted native checks, numerical-summary
matches, direct array/artifact comparisons, conditional results, and unverified
cases. Do not state that compilation or structural matching confirms numerical
accuracy, or describe the current 300 as unconditional element-wise agreement.
Include representative failure-to-confirmation case studies where evidence
supports them, with remaining limitations explicit.

## 7. Ontology and roofline analysis

`ontology/benchmarks/<bench>.yaml` is the logical benchmark source of truth;
indexes and similarity edges are derived. Read `ontology/README.md` and its
applicable generation instructions before changing ontology records. Review
inferred/automatically extracted classifications before relying on them for
research. Preserve their provenance and confidence. Do not relabel benchmark
features retrospectively to agree with framework rankings.

Join observations by benchmark ID and ontology revision. Separate intrinsic
workload properties (motif, precision, access pattern, synchronization needs)
from implementation properties (actual kernels, launches, traffic, layout,
resource usage, fusion, library selection). Record any source-to-port changes.

Time measurements alone do not establish roofline positioning. For floating-
point kernels collect work, memory traffic and its level, elapsed kernel time,
precision and operation conventions, plus matching hardware ceilings:

- Arithmetic intensity = floating-point work / bytes transferred.
- Achieved performance = floating-point work / kernel execution time.
- Record whether work/traffic are measured or analytically estimated.
- Profile CUDA, CUDA.jl, KA, and JACC separately; do not assume equal traffic
  or executed work. Preserve kernel mappings for fused/split implementations.
- Measure kernel-level positions; aggregate multi-kernel work/bytes/time only
  over a clearly defined region, never by averaging arithmetic intensities.
- Use separate unprofiled timing runs for performance and profiler runs for
  counters; record tool/version, commands, replay behavior, and permissions.
- If counters are unavailable, mark profiling pending/unsupported and label
  analytical estimates. Do not infer measured bottlenecks from ontology labels.
- Use appropriate bandwidth/latency/operation throughput for integer,
  cryptographic, atomic, or launch-dominated workloads instead of forcing
  every benchmark onto a floating-point roofline.

Candidate figures: success/feature coverage by motif, relative execution time
by motif/size, token/iteration distributions, error taxonomy by abstraction,
roofline positions for representative cases, and cost/performance trade-offs.
Use scientifically appropriate plotting tools with exportable figures and
regenerable scripts. Avoid treating ontology-similarity neighbors as independent
observations or independent train/test samples for any predictive recommender.

## 8. Fair comparisons, limitations, and performance protocol

Use protocol `five_trials_last_four_v1`: five execution trials, trial 1
excluded from the timing mean, trials 2–5 retained. Store run_count=5,
excluded_trial=1, retained_trials=[2,3,4,5], and the protocol ID in result
configurations/snapshots. This is the sole required timing protocol. Collect
one complete five-run set per benchmark/model/configuration, with no additional
measurement series. Generate runtime/performance tables and figures only from
complete, matched five-run sets.

Follow agent instructions §13: identical inputs, precision, tolerances, and
equivalent timed regions; both CUDA and CUDA.jl references; five trials with
trial 1 excluded from timing aggregates and mean(trial 2, trial 3, trial 4, trial 5) reported.
Correctness is required for all trials. Record seeds, reset state, JIT/setup
handling, device synchronization, hardware/software versions, flags, and all four
raw retained times. Report conditional comparisons and missing/failed data.
Four retained timings still provide limited variability evidence; do not call small
runtime differences significant without further justified measurements. Any
supplementary publication protocol must be specified, versioned, and applied
symmetrically while preserving the original five-trial records.

Performance tables and figures must compare **native CUDA (C++), CUDA.jl,
KA, and JACC**. Export matched per-benchmark/configuration rows for all four
models with all four retained trial times, mean, numerical status/evidence strength,
and conditional flags. Define native-relative speedup as
`t_native_cuda / t_model` and include Julia-to-Julia comparisons. Show matched
sample sizes and omitted/unsupported counts; keep conditional configurations
separate. Explain kernel versus end-to-end timing boundaries and exclude
compilation/JIT from kernel execution metrics. Keep any separately measured
compilation/startup costs available as different metrics. Never mix hardware,
input sizes, precision, or arithmetic flags to fill a four-model row.

Historical CUDA.jl token records are incomplete. Keep exact, estimated, partial,
and unknown counts separate; unknown is null, not zero. KA/JACC follow existing
CUDA.jl, and JACC follows KA, so their production costs are incremental and
order-dependent. Record what code/analysis each variant could see. Do not
present these as independent CUDA-to-abstraction costs. A separate controlled
subset with fresh contexts, matched inputs/budgets, and balanced order can
investigate independent translation effort without changing the production
KA-then-JACC workflow.

Keep translation, later optimization, verification repair, source migration,
shared infrastructure, profiling/analysis, and paper-writing effort separately
attributed. Do not charge writing tokens to a benchmark's translation or add
historical CUDA.jl estimates to new costs without labeling the distinction.
Measure as-translated implementations before any later tuning, retain both
versions, and give frameworks comparable tuning budgets. Separate pure
framework implementations from hybrids/vendor-library wrappers. One GPU vendor
supports claims about that tested hardware, not demonstrated cross-vendor
performance portability.

## 9. Versioned paper evidence and completion criteria

Raw transcripts/runs/databases remain under STATE. Keep concise observations,
protocols, claim records, analysis scripts, and evidence manifests in the
paper's `research/`. Each `research/snapshots/<id>/` identifies the baseline,
source/ontology/protocol revisions, selected result rows/run IDs, package and
hardware configuration, analysis command/code revision, and artifact paths or
checksums. Export required result subsets for reproducibility; a local DB path
alone is not a portable paper artifact. Mark unavailable data explicitly.

Generate tables and figures from a named snapshot, and tie manuscript numbers
to it. Update or invalidate affected claims when sources/results change. Never
hand-edit generated counts to agree with narrative. Record any LaTeX build
command/result and missing dependencies; unbuilt text must be labeled unbuilt.

A benchmark pair's paper contribution is recorded when both variant outcomes,
its observation and claim links (including a finding of no difference), result
paths, and current/pending section/figure work are in the combined checkpoint.
A publication claim is ready only when its supporting evidence and limitations
are traceable and its results can be regenerated. Unsupported outcomes are
valid observations, not successful translations.

## 10. Combined checkpoints and restart procedure

Use `research/checkpoints/checkpoint_<UTC timestamp>.md` and update
`research/checkpoints/LATEST.json` only after the checkpoint and referenced
records are saved. Use atomic replacement for the pointer. Keep previous
checkpoints; distinguish `inflight` from completed milestones. A saved in-flight
checkpoint is a valid restart record, not a declaration of completed work.

Every record must include:

1. Repository root, branch/HEAD, timestamp, protocol/instruction revisions or
   hashes, STATE path, source-layout/migration map, and baseline snapshot.
2. Current benchmark/variant/phase, source paths, per-variant checkpoint paths,
   finished and pending work, exact next commands, blockers, and running jobs.
3. Numerical status/evidence strength and conditions, both reference results,
   input/acceptance/error details, raw five-trial timings/mean or unknown,
   matched native CUDA/CUDA.jl/KA/JACC rows, token/iteration records and quality,
   compiler-attempt/check/repair counts and histories, unsupported features,
   and reproducible run/log/database locations.
4. Paper files edited, RQ1–RQ5 answer status, E1–E5 evidence/missing analysis,
   observation and claim IDs,
   supported versus tentative explanations, ontology revision, data snapshots,
   figure/table generation commands, citations, and manuscript build status.
5. Separate next translation and next writing actions with dependencies;
   section/figure/analysis backlog that must be handled before moving on.
6. Source, paper, and documentation commits/uncommitted files; unrelated dirty
   work to preserve; missing external artifacts and recovery instructions.

On restart: read the agent entry point, correctness file, this guide, LATEST.json
and its checkpoint; inspect live branch/HEAD/status, paths, DBs and jobs;
reconcile newer changes; then resume the named translation and writing actions.
If the pointer is absent/corrupt, inspect the newest complete saved checkpoints
and git history, recover a pointer without overwriting newer work, and document
the recovery. If no checkpoint exists, create a baseline checkpoint before
new implementation. Never rely on remote context compaction to recover progress.


## 11. Presentation: readable charts, common code, and combined workflow

The user prefers clean, compact, readable explanatory charts. Apply these
requirements to all paper figures and persist them in future checkpoints:

- Use consistent model identities across charts and diagrams: native CUDA
  dark gray (#333333, circle), CUDA.jl blue (#0072B2, square), KA teal (#009E73,
  triangle), JACC orange (#D55E00, diamond). Add markers/line styles so color
  is not the only distinction. The palette is defined in preamble.tex and
  research/analysis/figure_style.py; publication.mplstyle supplies plot defaults.
- Design at the actual publication size: approximately 3.35 inches for one
  column or 7 inches for two, readable 7–9 point labels, concise units, aligned
  panels, a shared legend where useful, and minimal grid/spine decoration.
  Preserve the ACM class's font sizes, margins, and spacing.
- Prefer compact distributions/ECDFs for the full population, selected paired
  plots and small multiples by ontology motif, and focused scaling/roofline
  panels. Avoid a 300-category bar chart, unreadable labels, 3-D decoration,
  rainbow palettes, and unnecessary dual axes. Use appropriate logarithmic
  axes and a clear parity/reference line for ratios; bar charts start at zero.
- Show denominators, unsupported/missing outcomes, conditional flags, and
  uncertainty appropriate to the evidence. Do not invent confidence intervals
  or treat four timing samples as four independent benchmarks. Tie captions
  to the RQ, metric, comparison set, and supported takeaway.
- Export vector PDF/SVG with embedded readable fonts; keep source data,
  snapshot IDs, and generating scripts. Inspect figures at publication size.
  Add ACM Description text for accessibility. No fabricated result charts.

Immediately after the introduction, introduce CUDA.jl, KA, and JACC through
one shared computation in native CUDA, CUDA.jl, KA, and JACC. The current
section 02 uses SAXPY (y[i] = a*x[i] + y[i]) in a two-by-two listing figure.
Keep assumptions, input/output semantics, precision, indexing conversion,
backend, and omitted setup/timing explicit. Preserve the same update direction
across panels. Treat snippets as illustrative until executed/validated at
pinned package versions; API documentation review alone is not execution.
Keep listings in snippets/ and their layout in figures/code_comparison.tex.

The methodology must include one combined vector diagram, currently
figures/agent_workflow.tex, showing reference inputs and contracts; the agent;
compiler/device-JIT evaluation; diagnostic-driven repairs; numerical checks;
ontology annotation/provenance; attempts/tokens/failure/evidence capture;
five-run timing and separate performance profiling; and RQ-driven analysis,
paper updates, and persistent checkpoints. Show feedback and data flows
explicitly, retain unsupported/failing outcomes in records, and distinguish
the protocol diagram from proof that all stages were executed.

The paper directory is a separate nested git repository. Record both the
translation repository and paper repository roots, branches, HEADs, and dirty
files in joint checkpoints. Run git/build commands from the appropriate root;
do not assume a source-repository commit includes manuscript/checkpoint files.

Build from 2026-PASC-Hecbench-Julia-paper/ with `make`; the ACM PDF is
build/main.pdf. Generated build products are ignored. Record the command,
LaTeX/package version, page count, warnings, and visual inspection in the
combined checkpoint. Replace draft metadata only with supplied author details
and official publication metadata; do not fabricate DOI/ISBN/rights information.

## Active cohort evidence refresh

Use `research/analysis/export_verified_study.py --source-root <translation-repo>` to regenerate matched result/configuration/hash/all-five-trial snapshots and portable compiler/token manifests. `research/data/adjudications.json` retains reviewed evidence invalidations or false alarms; no rawresult deletion. Regenerate figures from these CSVs only, then `make`. Exact sharedsession counters may exist even when pervariant attribution is unavailable: distinguish those scopes. Record namedsubkernel/precision configurations, reference corrections, algorithmadaptations, and excludedinvalidnativecases before aggregateperformance claims. Combined checkpoint tool lives at translation `tools/checkpoint_portable_study.py`; `LATEST.json` remains authoritative restart pointer.


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

### Omitted contract attestations

If otherwise complete immutable measurements omitted the top-level shared-input or equivalent-region attestation, use `reuse_missing_contract_attestations` with exact result/config hashes, source-boundary review and hashed input/timing evidence. Only omitted/null fields may become true; an explicit false cannot be repaired this way. Require all-four reuse, twenty valid checked JIT-excluded trials with the same explicit region, unchanged execution/configuration/tolerances/input contract, a positive serialized-input prefix and fresh saved-artifact comparisons. Preserve the original records and the resulting metadata adjudication; this is a reviewed descriptive correction, not a new GPU run or a repair of numerical failure.

### Canonicalized nondeterministic outputs

For an operation with legitimately unordered append/reduction results or tied minimizers, retain the complete raw outputs and compare an independently validated complete canonicalized set/multiset or minimizer-membership contract. Use `numerical_agreement_kind: property_equivalence_not_identical_raw_values`, with `numerical_agreement_claim: false` and `raw_output_identity_claim: false`; exact canonicalized artifacts do not establish identical raw numerical outputs. Preserve multiplicities, candidate IDs/parameters, thresholds, input bits and all guards; document what is and is not invariant. Do not serialize the native operation to manufacture ordering or require one arbitrary tied winner. Count these as qualified accepted property contracts, separately from the requested exact four-model numerical target and from scientifically faithful application validation.
