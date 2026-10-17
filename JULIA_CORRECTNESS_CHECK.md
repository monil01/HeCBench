# Julia Correctness Check

## Active expansion: all 300 benchmarks and scientific interpretation (2026-10-17)

**This section is the current authority.** It supersedes historical 100-pair completion/stop instructions, earlier feasibility-subset targets, and any instruction to commit `paper/research/`. The user has explicitly reopened translation and paper work, requested multiagents, and expanded coverage to the entire frozen 300-benchmark population. Read the other two canonical guides and the newest combined checkpoint. The previous 100 pairs/200 portable implementations/107 configurations remain valid evidence, not results for 300.

### Population, ownership, and completion

- Frozen population: `.porting-state/portable_julia/baseline_300.json`. Active census: `batch_300.json`; worker queues: `batch_300_{maps_ports,portable_infrastructure,stencil_ports}.json`. Do not replace difficult benchmarks with names outside the frozen 300. Re-examine the pilot's rejected candidates and preserve the rejection history.
- Target all 300 benchmark names in native CUDA, CUDA.jl, KA, and JACC: 600 portable versions in addition to the CUDA.jl references. Process one benchmark KA then JACC; three workers own disjoint queues and the coordinator owns shared tooling, source commits, ontology aggregation, and manuscript integration. All GPU timings remain serialized on GPU6 under `/tmp/hecbench-julia-study-gpu.lock`.
- Historical CUDA.jl PASS labels are not modern certification. Re-audit semantics, inputs, precision, reference accuracy, and timing region; preserve original source and account for reference repairs separately. Retain the completed pilot evidence where its contract/source/environment still match.
- Maintain the full denominator 300 in coverage tables, with separate model-specific counts: not started, in progress, verified, conditionally verified, failed, reference blocked, environment/hardware blocked, and evidence-backed feature unsupported. A stub, timeout without diagnosis, or unfinished port is not unsupported. Show library interoperability and scoped kernels separately from translated device kernels.
- Work continues through all queues and manuscript analysis. Never claim 300 successful benchmark pairs or 600 verified portable versions from target counts. If a genuine capability/hardware barrier persists after investigation, retain its minimal reproducer, logs, exact versions, and scoped outcome. Do not conceal it by replacing the benchmark or weakening equivalence.


**Scoped multi-GPU exception for the frozen ring benchmark:** the coordinator may use the otherwise idle physical devices5 and6 with `gpu: "5,6"` for a genuine two-device exchange. Retain the global GPU6 study lock, check device5 is idle before starting, and stop/defer if another workload owns it; never interfere with device0 or other users. Record both UUIDs, visibility/order, peer-copy semantics, synchronization, precision and scope. This is a separate two-GPU stratum, excluded from single-V100 aggregate rankings. KA/JACC host CUDA orchestration and zero-device-kernel interoperability must be labeled honestly. A one-device self-copy is diagnostic only and never satisfies multi-GPU correctness/performance coverage. Other benchmarks retain GPU6-only measurement.

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

Record translation candidates, compiler checks, feedback repairs, runtime/numerical repairs, correctness, performance, and token usage separately per model and phase. Start usage boundaries before new work when available; shared orchestration remains shared. Missing exact consumption stays null and approximate records must state method and coverage. Never compare total costs using heterogeneous source-size proxies.

Prospective usage measurement is now available through `tools/capture_translation_usage.py`, using registered cumulative harness streams for this root and its three workers. Before each untouched benchmark, record a shared `benchmark_workflow` begin; capture CUDA.jl reference work and KA/JACC translation begin/end windows separately. A begin must occur in a separate tool/model round **before drafting that phase's code**; close after the verified or diagnosed outcome. Do not backfill started work. Record exact usage-event timestamps, input/cached/output/reasoning counter deltas and marker lag; these are observed activity-window counts, not exclusive billed translation costs. Repeated context and coordination are included. Disclose interleaved benchmark work; overlapping workflow/subphase spans cannot be summed. Preserve earlier partial source proxies and unknown history alongside these prospective records, and report measurement-method coverage separately. Verify the identity of every profiled kernel; initialization kernels do not characterize the scientific operator. Profiling/filter repairs remain shared measurement effort, not target translation failures.

Only the coordinator stages source commits; preserve one separate commit per newly verified KA/JACC version and distinguish reference/infrastructure/tuning commits. Workers report ready result/config paths and do not stage another worker's files. Never commit `.porting-state`, research evidence, binaries, or unrelated work by broad staging.


**Native numerical-reference repairs:** a native/CPU disagreement first needs a minimal witness and a source-grounded mathematical contract; do not assume either reference is correct. If a native defect is independently established, a corrected study kernel is authorized only with the original source and mismatch preserved, exact correction/source hashes, independent numerical evidence, and separate native-repair effort/failure records. Set `native_reference_repaired: true` and a `reference_provenance` object in the configuration. Classify results as conditionally verified against the corrected reference, not verified against untouched native CUDA; exclude corrected-native timings from untouched-native aggregate charts and show the conditional stratum separately. Do not silently change a physical model, regularization choice, tolerance, or scientific question to manufacture agreement.

**Reproduced native behavior with an established scientific defect:** if an independent witness establishes a semantic defect but the study intentionally preserves the original native behavior, set `native_behavior_known_scientific_defect: true`, `exclude_from_scientific_fidelity_aggregates: true`, and explicit witness/reference provenance. Report `verified_native_behavior_known_scientific_defect`, separately from both ordinary named verification and corrected-native conditional verification. Such evidence confirms reproduction of executable behavior, not the intended scientific algorithm. Keep its timings in a separately labeled diagnostic stratum; exclude it from faithful-operator aggregate charts and abstraction recommendations. Retain a justified correction as a separate repair task, with original outcomes preserved. If classification follows execution, use a classification-only `src/<bench>-julia/study_adjudications.json` supplement bound to the immutable result/config hashes and loaded by `tools/study_evidence.py`. Do not mutate archived configurations or rerun just to change a descriptive label; supplements cannot change inputs, commands, tolerances, trial data, or pass/fail results.

### PASC paper and persistent restart

The manuscript must engage domain scientists: begin with scientific decisions, why a 300-workload corpus is useful, and what transferable evidence a simulation or analysis developer gains. Present scientific workload categories and numerical requirements before implementation taxonomy and aggregate speedups. Explain limits of kernel proxies for whole applications and of one NVIDIA platform for portability. Do not promise community artifacts that are not actually distributable.

Keep RQ1–RQ5 but orient them to (1) scientific-family coverage and fidelity; (2) model-specific compiler/semantic failures and effort; (3) paired performance by family and scale; (4) scientific-intent and model-specific ontology mechanisms; and (5) actionable abstraction choices for scientific use cases. Show representative scientific case studies and open hypotheses, with readable compact figures, per-model taxonomy panels, coverage/unknown denominators, and evidence-backed recommendations. Keep the common four-language example and combined methodology diagram. Distinguish the census target from current completed results in abstract, tables, and conclusions.

`2026-PASC-Hecbench-Julia-paper/research/` is **local-only and ignored**. Keep data, scripts, observations, and joint checkpoints there for session restart; do not force-add them to the paper remote. Only LaTeX, references, snippets, final publication figures, generated LaTeX summaries, and manuscript documentation belong in Overleaf sync. Local research is not a published archive. Export a small checkpoint pointer/summary to the source repository without raw artifacts. Checkpoints record both repository HEADs, all300 status/model counts, ownership, versioned ontology/taxonomy schema, source and paper progress, running jobs, and exact next actions. Historical complete100 checkpoints do not stop this expansion.

Correctness priority: report each model's strengthened evidence separately from its historical PASS. Explicitly audit scientific boundary conditions, units/scaling, precision/contraction, reduction ordering, iterative convergence, RNG contracts, and complete output coverage when applicable. No accuracy claim transfers automatically from CUDA.jl to KA/JACC or from one dtype/size to another.

---


Status date: 2026-10-16

## Restart links

Start with [AGENT_INSTRUCTIONS_CODEX_Julia.md](AGENT_INSTRUCTIONS_CODEX_Julia.md),
then read this file and [paper_writing.md](paper_writing.md). Follow the
[combined checkpoint pointer](2026-PASC-Hecbench-Julia-paper/research/checkpoints/LATEST.json)
to resume both the current translation/verification task and its pending paper
observation/analysis. Database counts below describe recorded results, not a
new verification run performed by this documentation update.

## Scope

This file summarizes the numerical correctness sweep and subsequent corrections for Julia
(`CUDA.jl`) ports in HeCBench.

Input set:
- `441` Julia ports with existing `ok` status in `.porting-state/coverage.db`
- CUDA was treated as the reference model
- Results were written by `tools/verify_numeric.py`

Tracking artifacts:
- `.porting-state/numeric_accuracy.db`
- `.porting-state/numeric_accuracy/reports/counts.txt`
- `.porting-state/numeric_accuracy/reports/mismatches.txt`
- `.porting-state/numeric_accuracy/reports/unverified.txt`
- `.porting-state/numeric_accuracy/reports/not_run.txt`
- `.porting-state/logs/numeric_accuracy_summary_2026-09-24.md`

## Results

```text
pass|native_pass_trusted|259
pass|numeric_stdout|41
mismatch|numeric_stdout|72
unverified|no_numeric_data|56
not_run|reference_fail|1
not_run|run_status|11
not_run|timeout|1
```

Summary:
- `300 / 441` passed either benchmark-native correctness checks or numeric
  stdout comparison.
- `72 / 441` reported mismatches and require investigation.
- `56 / 441` did not expose comparable numeric data and need benchmark-specific
  metadata, checksums, or artifact comparisons.
- `13 / 441` did not complete a CUDA/Julia numerical comparison during the
  sweep.

## Verification strength of the 300 passing CUDA.jl records

All 300 have a passing record in the numerical verification database under
their recorded conditions: 259 trust native verifiers and 41 compare numerical
stdout summaries. These checks do not establish identical-input, element-wise
CUDA equivalence for all 300. Native verifier strength varies, summaries may
omit outputs, and `atan2` has a conditional pass requiring CUDA `--fmad=false`
(see below). Thus the baseline is 299 other passing records plus one explicitly
conditional record, not 300 unconditional raw-output comparisons.

## Interpretation

`pass|native_pass_trusted` means CUDA and Julia both emitted benchmark-native
`PASS` verdicts and neither emitted `FAIL`.

`pass|numeric_stdout` means comparable non-timing numeric stdout values matched
within the verifier tolerance.

`mismatch|numeric_stdout` means comparable numeric stdout was found but CUDA and
Julia values differed beyond tolerance.

`not_run|reference_fail` means the CUDA reference emitted a native `FAIL`
verdict, so the Julia result cannot be compared against a passing CUDA
reference.

`unverified|no_numeric_data` means the benchmark ran but did not expose
comparable non-timing numeric output. These ports need benchmark-specific
numeric metadata or artifact comparison before numerical equivalence can be
claimed.

`not_run|run_status` and `not_run|timeout` mean the numerical sweep could not
complete the CUDA/Julia pair for that benchmark.

## Correction: atan2 (2026-10-16)

Fixed the Julia CPU-reference argument order for float32, int32, and int16
outputs to match the GPU launches and CUDA reference. Native verification now
checks all three output types and propagates failure through the process exit
status. The prior float-only threshold of 10 has been tightened to 1e-3;
integer RMSE must be zero.

Verification used Tesla V100 (sm_70), the local CUDA 12.0 toolkit, and canonical
CUDA arguments `10000000 100`. With CUDA `EXTRA_CFLAGS=--fmad=false`, both
implementations printed zero RMSE for all three output types; the numerical
verifier reported `PASS numeric_stdout` with zero absolute/relative difference,
and coverage reported `MATCH`. Numerical comparison tolerances were unchanged.
This compares the printed RMSE summaries rather than raw output arrays.

The default CUDA build with FMA enabled reported int32 RMSE 1.177096 and int16
RMSE 0.000548 while Julia reported zero. The passing record therefore applies
to the controlled build with contraction disabled. Reproducing this record
requires that build flag; default-build summary matching remains unresolved.
The original input generators still differ between CUDA and Julia.

Evidence, build commands, correction token accounting, and the local checkpoint
are under `.porting-state/numeric_accuracy/fixes/atan2/`. Original translation
usage and attribution are preserved.

## Next Work

1. Investigate the `72` mismatch rows first.
2. Add `numeric_check:` metadata or deterministic checksums for the `56`
   unverified rows.
3. Repair or retest the `13` not-run/reference-blocked rows.
4. Rerun `tools/verify_numeric.py` after each fix and update the local summary.


## Instructions for KA and JACC variants

The user-authorized target is two additional implementations of the frozen
300-benchmark CUDA.jl passing set (600 variants). The authoritative workflow
is §13 of [AGENT_INSTRUCTIONS_CODEX_Julia.md](AGENT_INSTRUCTIONS_CODEX_Julia.md).

### Naming and progress

Use one parent `src/<bench>-julia/` containing these implementation directories:

- `<bench>-julia-cuda.jl/`: CUDA.jl (`julia` model ID).
- `<bench>-julia-KA/`: KernelAbstractions.jl (`julia-ka` model ID).
- `<bench>-julia-JACC/`: JACC.jl (`julia-jacc` model ID).

The CUDA C++ reference remains `src/<bench>-cuda/`. Existing flat CUDA.jl
sources remain valid during the documented transition; inspect actual paths
and the source-layout map. Migrate per benchmark only after adapting discovery,
execution, and relative paths, preserving source contents and provenance.
Instruction edits alone do not move or reverify source files.
Freeze the benchmark list and reference conditions under
`.porting-state/portable_julia/baseline_300.{json,txt}` before translation.

This document update adds instructions; KA/JACC implementation counts have
not yet been audited. No new KA/JACC passes are claimed here. Subsequent
reports must show independent per-variant counts for pass, conditional pass,
unsupported feature, missing input/dependency, build/run failure, numerical
mismatch, reference disagreement, unverified, and not attempted. Include a
breakdown by verification strength, with the snapshot as the denominator.

### Correctness and execution-time records

For each variant, compare against **both CUDA and CUDA.jl**, and confirm the
references agree under the recorded conditions. Use identical inputs, sizes,
precision, seeds, and documented tolerances; record any remaining input or
reference limitation. Preserve native checks, compare numerical outputs or
artifacts when available, and label summary/native-only evidence explicitly.
A structural stdout match alone does not establish numerical correctness.
Keep the `atan2` build condition visible rather than promoting it to an
unconditional pass.

Use only protocol `five_trials_last_four_v1` for all four implementations:
one complete five-run measurement set per benchmark/configuration, with run 1's
time excluded and runs 2–5 averaged. No additional measurement series is
required. Use only complete, matched five-run sets in performance analysis.

Run each available implementation (`cuda`, `julia`, `julia-ka`, `julia-jacc`)
five times per benchmark/configuration. Trial 1 is warm-up: retain its
correctness result and timing, but exclude its time from the aggregate.
Report trials 2, 3, 4, and 5 and their arithmetic mean in microseconds. Verify
correctness in all five trials, reset state between them, reuse the process
where possible to avoid repeated JIT, synchronize GPU timing boundaries, and
match the CUDA execution region. Document necessary process/JIT deviations.
Run timed trials sequentially on the same device. Record failed trials and
missing timing as statuses/nulls; do not report valid aggregate performance
for a failing variant.

Store variant/reference/trial results in
`.porting-state/portable_julia/verification.db`, raw timings in
`.porting-state/portable_julia/timings.csv`, and retained-run means in a summary
CSV. Include commands, compiler flags, package versions, GPU/backend,
configuration IDs, reference revisions, and log paths. Existing two-model
verification and performance tools require extension before they can produce
these records. Preserve the original CUDA.jl verification database.

### Evidence for numerical accuracy and compiler-in-the-loop attempts

The paper must explain **how numerical accuracy was confirmed for every
benchmark/version**. Follow agent instructions §§6.2–6.3: link each result to
its source/attempt/configuration/trial, input-generation details or input hashes,
precision, both reference commands/revisions, native checker or compared output,
acceptance rule/rtol/atol, checked values/elements, error metrics, nonfinite and
shape handling, raw logs/artifacts, and any conditional build requirement.
Record native-check, numerical-summary, and array/artifact evidence separately;
PASS text or successful compilation alone does not establish full equivalence.
Preserve the existing 300 records' actual evidence strength and historical gaps.

For CUDA.jl, KA, and JACC separately, maintain `translation_attempts.jsonl` and
manifest counts of candidate attempts, compiler/build/JIT checks, compiler-
feedback repair rounds, runtime/numerical repairs, and the attempt reaching
first compilation and first verified success. Include the initial candidate,
failed attempts, diagnostics, resulting edits, and associated tokens. Track
original translation and later corrections independently. A no-op Julia build
is not a successful JIT compilation check. Repeated checks on unchanged source
and the five timing trials are not new translation candidates. Record missing
historical counts as null with accounting provenance; native CUDA translation
attempt counts are not applicable to this study.

### Four-implementation performance analysis

The comparison is **native CUDA (C++), CUDA.jl, KA, and JACC**. Produce a row
for each matched benchmark/input/precision/configuration with all four models'
trial 2, 3, 4, and 5 execution times, their mean, correctness status, evidence
strength, and build conditions. Retain trial 1 and its correctness verdict,
excluding only its timing from the mean. Define speedup versus native CUDA as
`t_native_cuda / t_model`; include comparisons between the Julia abstractions.
Preserve equivalent timed regions and report compile/JIT/startup/end-to-end
metrics separately when collected. Use correctness-qualified matched sets for
aggregates; show denominator, unsupported/missing cases, and conditional cases.

### Translation consumption and unsupported features

For every benchmark/framework pair, record translation tokens separately in
`.porting-state/porting_logs/<bench>-julia-ka/` or
`.porting-state/porting_logs/<bench>-julia-jacc/`. Include input/output/cached/
total tokens, per-iteration records, method, sources, completeness, and
limitations. Include failed/deferred work; preserve the original CUDA.jl costs.
Use exact usage when available, clearly labeled estimates otherwise, and
`null` when unknown. Summary-only estimates remain partial.

Record unsupported features per benchmark and framework in the manifest and
`.porting-state/portable_julia/unsupported_features.jsonl`. Include the required
feature/source location, backend and package version, supporting API/source
or failure evidence, attempted workaround, portability/performance effects,
and tokens spent. Distinguish unsupported APIs from dependency/input failures
and numerical bugs. Identify any CUDA-specific fallback or hybrid dependency;
it cannot silently count as a pure portable implementation.

After each batch, update this report with actual per-framework correctness
counts, conditional passes, timing coverage, token-accounting coverage, and
unsupported-feature benchmark lists. Link the evidence and local checkpoints.


### Paper integration and checkpoint completion

The manuscript destination is `2026-PASC-Hecbench-Julia-paper/`, with each
section in its own TeX file. Follow [paper_writing.md](paper_writing.md).
Complete KA then JACC per benchmark. Link each completed pair's correctness,
five-trial performance, tokens/iterations, and failure/unsupported records to
its ontology ID and paper observations. Update claims only to the strength of
the available evidence; distinguish inferred resource labels from measured
roofline data and retain conditional results.

After each variant, pair, or paper batch, save both per-variant progress and a
combined translation/paper checkpoint, then update
`2026-PASC-Hecbench-Julia-paper/research/checkpoints/LATEST.json`. Include paper
section edits, compiler-attempt histories/counts, numerical acceptance/evidence,
the matched four-model timing data, evidence/figure snapshots, unresolved
hypotheses, and exact next translation and writing actions. A verification checkpoint without paper state
is incomplete. See agent instructions §14 for the restart record contract.


### Research-question evidence mapping

Follow RQ1–RQ5 and E1–E5 in [paper_writing.md](paper_writing.md) §3. This report
provides RQ1/E1 numerical correctness and feature-coverage evidence; link
compiler-attempt/failure histories to RQ2/E2, the matched native CUDA/CUDA.jl/
KA/JACC five-run timing data to RQ3/E3, and ontology/profile comparisons to
RQ4/E4. Feed supported results and their limits into RQ5/E5 abstraction-selection
guidance. Decision synthesis is not a new experiment unless validation runs
were actually performed.

Tag observations, claims, and analysis snapshots with `rq_ids` and
`experiment_ids`. Update the research/protocol.md RQ answer matrix and combined
checkpoint with available evidence, missing analysis, answer status, and paper
locations. Numerical pass counts alone do not answer performance, mechanism,
or abstraction-selection questions. Keep conditional/weak evidence and
unanswered questions explicit.


### Paper presentation and accuracy of examples

The active ACM sigconf manuscript is
`2026-PASC-Hecbench-Julia-paper/main.tex`, with separate section files. Its
post-introduction code panels illustrate the same SAXPY operation in native
CUDA, CUDA.jl, KA, and JACC; illustrative excerpts are not additional verified
benchmark results. The methodology diagram connects compiler/JIT feedback,
numerical verification, ontology recording, cost/failure evidence, matched
five-run performance, profiling, and persistent paper checkpoints. Follow
paper_writing.md §11 for readable, compact charts and evidence-backed captions.
No new measurement or correctness status is implied by manuscript conversion.

## Active KA/JACC cohort status (generated checkpoint summary)

At 2026-10-16T23:02Z, **20/100 benchmark pairs** have passing five-trial full-output comparisons against BOTH native CUDA and CUDA.jl; the local runner currently contains20 matchedconfigurations. Allfive trials correct, meanlastfour, serialized GPU6TeslaV100. These are named measured configurations, not claims of every originalCLI/precision/subkernel case. Source includes separate variantcommits and verification.json; paper research/data/matched_results.{json,csv} contains alltimings and checks.

- Historical CUDA.jl300-pass baseline remains frozen (259native-verdict,41numericstdout); 87source-pattern flags requiresemanticreview and do not imply87confirmedfailures.
- Variant coverage: KA20, JACC20 for current matchedcohort, fullbinary/scalaroutputchecks with pervariantCPUoracles; originalhistoricaldatabase unchanged.
- Adaptations/subsets: atomicAggregate uses disclosedsharedmemory matching; atomicPerf onlyvalidglobal9cases with nativeinvalidsharedcases recorded; allpairs k3only; flipFP32only; bitpermute lg15only. OriginalCUDA.jl source preserved; atomicCAS reference genuinely repaired separately.
- Runtime: full5series permodel, retainedoutliers visible (heat2dJACC,hotspotKA), nopreferredreruns. Reused verifiedreferencebundles are hash-checked and tagged.
- Costs: pervariant source/summaryproxies partial, unknownactual counts null. Exactparent sessioncounters are mixedsharedwork, not perbenchmark attributable. Compilerchecks/repairs separatefromtimingtrials.
- Unsupportedfeatures: no adjudicated framework-unsupportedresult in current20pairs. This doesnot establish coverageoutside these cases. Deferredreference disagreements andenvironmentfailures remain distinct.
- Paper: realdata figures, RQ1–RQ5 results evolving; exploratoryreversecounterprofiles all4 collected; no floating-point measuredroofline yet. Resume via paper research/checkpoints/LATEST.json.


### Active study resource-policy checkpoint (2026-10-16)

For kernels without shared data, explicitly set JACC `launch_spec(shmem_size=0, ...)`. Installed JACC1.4.0 uses automatic maximum dynamic shared allocation when omitted; a controlled triad experiment measured49152B/block and a shared-memory occupancy limit2, versus0B/limit32 after this change, with unchanged geometry/registers and allfive fulloutput checks passing. Kernels requiring shared arrays must request their actual required bytes, never blindly zero them. Treat this as a measured configuration-specific resource-policy finding, not universal framework overhead.

Keep original verified default-resource source/results as an ablation. A changed launch-resource policy is a separately named `performance_tuning` configuration, with tuning attempts/tokens separate from original translation. Reuse unchanged validated reference/KA bundles under exact source, command, input and protocol checks. Any nonsemantic dependency exception requires explicit old/new SHA256 and a reviewed reason; the current exception only adds Int64 artifact typing after timing, leaving Float32/64/Int32 paths unchanged. Rawfive samples and allcorrectness verdicts remain retained. The paper's `research/data/configuration_dispositions.json` selects primary configurations without counting old/new resource-policy versions as independent benchmarks.


## Completed 100-pair study (2026-10-17 UTC)

The requested cohort is complete: **100 benchmarks, 200 separately committed KA/JACC versions, and 107 primary matched configurations**. The five-trial protocol excludes trial 1 and averages trials 2–5 for native CUDA, CUDA.jl, KA, and JACC. All recorded numerical comparisons passed; final numerical, effort, and source-proof audits are in `2026-PASC-Hecbench-Julia-paper/research/review/`. Seventeen additional resource-policy configurations are retained as ablations.

Resume by reading `AGENT_INSTRUCTIONS_CODEX_Julia.md`, `JULIA_CORRECTNESS_CHECK.md`, and `paper_writing.md`, then following `2026-PASC-Hecbench-Julia-paper/research/checkpoints/LATEST.json`. A checkpoint marked `complete_100_pairs_manuscript_review_ready` supersedes historical active-work notes. Do not restart the completed cohort or launch new measurements automatically.

The finished technical manuscript and build instructions are in `2026-PASC-Hecbench-Julia-paper/README.md`; the PDF is `build/main.pdf` inside that repository. Author/publication metadata and an archival artifact release remain submission tasks. The study covers named configurations on one V100, not all CLI modes or all 300 historical ports. Ninety-nine pairs contain translated kernels; MRG32k3a is explicitly a cuRAND interoperability case. Numerical evidence distinguishes complete arrays, defined struct payload, canonicalized multisets, and complex property-score checks.

Per-version compiler checks, source candidates, known repairs, correctness, timing, feature outcomes, and ontology records are preserved. Token records are partial estimates/proxies; exact consumed or billed tokens per version are unavailable and must not be inferred from source size or shared session counters. Missing repair counts remain unknown. Raw numerical artifacts and detailed local logs remain in `.porting-state`; portable results, provenance, audit summaries, and analysis scripts remain in the local ignored research folder; publication files are tracked in the paper repository.
