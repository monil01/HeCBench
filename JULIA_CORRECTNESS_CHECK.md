# Julia Correctness Check

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

Per-version compiler checks, source candidates, known repairs, correctness, timing, feature outcomes, and ontology records are preserved. Token records are partial estimates/proxies; exact consumed or billed tokens per version are unavailable and must not be inferred from source size or shared session counters. Missing repair counts remain unknown. Raw numerical artifacts and detailed local logs remain in `.porting-state`; portable results, provenance, audit summaries, and analysis scripts are committed in the paper repository.
