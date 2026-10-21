# Private CPU capture adapter proposal: allreduce

## Scope and authorization

This is authored source only. The adapter, imported parser and full-artifact checker
have **not been run or imported** by this worker. No GPU, builds, numerical runs or
tests occurred. GPU5 remains unauthorized. A root CPU-method review is distinct
from the user's prospective two-device execution permission. Neither is inferred
from old accepted numerical results.

The method identifier is `allreduce_two_device_rank1_private_add_NCU2022_4_1_v1`.
It is a **separate two-device method**: the original two MPI ranks still execute
all six cases, warmup and five full checks, while Nsight wraps rank1 only and a
source marker enables one local private add in retained trial2, n=1,048,576,
first repeat. Sampling geometry is32 blocks×256 threads,524,288 addends. It is not
an allreduce communication roofline, whole-ROI placement, collective timing,
application-replay profile, or a single-device replacement. The added synchronous
marker and profiler overhead mean new elapsed STUDY_TRIAL fields are diagnostic
only; original five-trial numerical/performance results remain immutable.

## Adapter interface (root adoption required)

`adapt_capture.py --review REVIEW.json --review-sha256 SHA --capture-root RAW
--output-root FRESH_TMP_BATCH --execute-cpu-adapter`

The review must be frozen **before** prospective capture, with exact final runner,
proposal, parser, old2022 NCU ELF, canonical collector reference, and old2022GPU6
calibration-review bindings. It must declare `root_reviewed_for_cpu_adaptation:true`
and exact adapter SHA. Those parser/calibration/source hashes must also be in the
runner's actual prospective `all_readonly_bindings`, including every calibration
supporting file. Root should archive that review with the authorization receipt.
Require actual `readonly_bindings_before/after` and `snapshot_bindings_before/after` maps, plus lossless source snapshots created before target execution. No2025 fallback, opportunistic parser replacement or retroactive target-source
reconstruction is permitted. `adapter_contract.template.json` is deliberately
unapproved and must be regenerated after the other agent freezes v2.

The adapter reads all four terminal raw model casefiles and inventories **every**
retained raw file before and after conversion. The original capture tree is never
written. Fresh output has `allreduce/{cuda,cuda_jl,ka,jacc}/` directories. Each gets:

- Standard `plan.json`, `profile_provenance.json`, raw `counters.csv`,
  `execution.log`, actual custom `collector_source.py`, exact `parser_source.py`.
- `parent_profile_provenance.json`, `adapter_source.py`, and
  `allreduce_two_device_review.json` with lossless base64 small raw logs/JSON,
  original-plan identity, parent provenance, rank/session registries and full
  original raw-file hash/size inventory. Large full captures stay at original
  retained paths; these are not checksums substituted for full numerical checks.
- `derived.json` **only** after every method/counter/correctness check passes.
  Any failure preserves available raw fragments and a failure provenance with no
  derived record/position. Failed/missing captures are never synthesized as zero.

`collector_sha256` identifies the actual custom runner. The standard collector's
source is a separately reviewed policy reference, never falsely attributed as the
executed collector. `new_gpu_processes:0` and `new_build_processes:0` describe only
this CPU adaptation; original process/build counts remain in `profile`/`builds`
and archived parent receipts. This is not a CPU recovery of a standard collector
run and must not use its narrower `CPU_only_recovery` proof branch.

## Mandatory joins and numeric semantics

1. Raw slot plan SHA→captured `staging_plan_sha256`; captured proposal SHA→reviewed
   proposal; selected result/config bytes→exact hashes/CID; model directory→model.
2. Final custom completion is necessary but insufficient. Require profile rc0,
   no guard/timeout/foreign stop, own-session quiescence, unchanged contemporaneous
   source bindings and all actual successful build receipts. Unknown ancestry or
   source identity does not become a pass through adaptation.
3. Explicit root-recorded userGPU5 permission bound to final proposal/runner,
   both physical UUIDs and four models. Both-device initial/final idle snapshots
   and all monitored GPU observations are retained. Exact registered rank0 maps
   to physicalGPU5; exact registered rank1/profiler map to physicalGPU6. The latter
   uses raw counter ProcessID plus three PCI counters and NVML UUID→PCI mapping;
   source markers additionally attest both ranks before warmup and after all5checks, matching actualPID/logicaldevice/PCI/UUID against registry and contemporaneous snapshots. Registry SID/UID/token/
   startticks/target argv are retained, not replaced with current `/proc` facts.
4. Exactly five original ordered STUDY_TRIAL records, full checks, original region,
   jit_excluded,18 source repeats and complete5,269,603Float32 values per trial.
   Recompute every word from the independent dyadic formula/receiver-chunk layout/
   input-prefix/guards, and hash-join **each trial** to both unchanged CUDA and
   CUDA.jl full saved artifacts. All20 captures mandatory; metadata certificates
   alone are insufficient.
5. Actual begin/end markers: rank1, trial2, n1M, repeat1, offset0, count524288,
   actual registered rank1PID; exactly one authored add. Both final guards assert
   original96 total ring-add calls, expected candidate/enter/exit counts. Counter
   raw group must be exactly one intended function/grid/block and rank1PID/PCI6.
6. Complete **27 standard metrics plus3PCI plus2compute-capability attributes** requested and present as finite,
   nonnegative numbers. CSV metadata strings are retained, not parsed as metrics.
   No unknown required counter/unit and no partial payload can yield a position.
   Require positive duration; unexpected tensor work holds conversion pending a
   separate typed review. Packed-half convention remains physical two lanes from
   exact old2022GPU6 review, not useful-work reinterpretation.
7. Use the unchanged frozen `read_csv`/`derive` functions. Preserve raw counts,
   units, formula and thresholds. Root's regular audit recomputes each saved record
   with exact parser bytes. No invented empirical sustained peak, wholeROI,
   communication cost or causal bottleneck; `causal_bottleneck` stays unknown.

## Required shared adoption changes (proposal only; no edits made)

**Freeze allowlist** in summarize must add `adapter_source.py` and
`allreduce_two_device_review.json`. Existing parent-provenance and raw/parser/
collector filenames are already retained. Also freeze the prospective reviewed
method JSON and calibration review through the envelope. Root should preserve
original raw inventory/adapter receipt under a small completion-review namespace.
Large GPU captures remain private with exact paths/hashes as already permitted.

**Audit branch** must recognize ONLY `allreduce_two_device_method` with exact
method ID, benchmarkallreduce and four canonical models. Reject if absent reviewed
method authority. Validate archiveSHA, adapterSHA, parent provenanceSHA, original
planSHA, source/config/resultCIDs, explicit two-device authorization/UUIDs, both
actual rank process mappings, all5 full correctness/marker checks and complete32
raw metrics. Recheck available full artifacts and dual-reference artifacts using
archived paths/hashes; if unavailable, mark audit qualification or failed required
retention honestly, never claim fresh revalidation. Retained base64 small raw
entries must individually reproduce SHA/size and exactly match parent evidence.
Keep generic mathematical `derive` recomputation unchanged. Do not mistake this
for the `profile_window` branch currently restricted to RayleighBenardConvection.

**Summarizer selection/reporting**: custom success may become standard
`profile_process_completed` ONLY after this branch is explicitly adopted/validated.
Never feed custom `profile_completed_all5checks_and_GPU6identity_verified` directly
as accepted status. Source ROI authority must bind actual selected derivation;
rank1 add membership is proved by source+markers, but no whole elapsed ROI claim.
The existing generic fallback string `partial_first_specialization_geometry` is
inaccurate for this method: add a method-specific `single_rank1_private_add_sample`
ROI_coverage and a measurement_method column/qualifier. Preserve old blocked/
failed attempts and original numerical qualification=`multidevice`.

**Ontology/publication/checkpoints**: retain allreduce per-benchmark and all4model
assessment slots; link selected method evidence to each implementation without
relabeling its mixed kernel/communication algorithm as pure kernel. Publish one
rank1-add geometric projection per implementation only, with two-device execution,
physicalGPU6 measurement, cache-all/kernel-replay and instrumentation covariates.
Do not multiply by ranks, sixsizes, repetitions, dispatch counts, or run time.
Do not claim collective hardware coverage from a local-add sample. Original
performance comparison remains the old accepted five trials, mean2–5; this new
profiling is a separate instrumented method. Checkpoint must distinguish
`method_prepared_not_authorized`, `capture_failed`, `CPU_adapter_rejected`,
`capture_adapted_pending_root_audit`, and `root_audited_rank1_sample`, independently
from numerical translation status and empirical bottleneck/causal unknowns.

## Current proposal issues / review gates

At first read the proposal omitted tensor metrics and utilization percentages,
used float conversion on every raw metric including string metadata, and used60s
combined lockwait. The other agent is revising these; this adapter never silently
repairs completed incomplete counters. A new reviewed prospective version is needed.
The profiler wrapper identity, actual rank mapping, long idle/lock wait and
source-bound calibration are runner responsibilities, not permission to backfill
from today's process/device state.

The proposed v2 runtime label is `ALLREDUCE_DEVICE_ATTEST` with fields
`rank,logical_device,pid,pci,uuid,phase`: exactly4 records, each rank before its
first warmup add and after all5 checks. The adapter requires these actual records
and joins all fields to the captured rank registry and initial physical-device
snapshot; registry expectedGPU5/6 PCI/UUID environment values must match too. Every prospective source snapshot is decoded/rehashed against its original binding, with equal before/after snapshot maps.
The original v1 only probed rank1; it must not be adapted under this stronger
method. No v2 runtime observation or successful adapter execution is claimed.
