# PRIVATE tmpfs evidence integration v2 — root review required

This packet corrects two source-review blockers in immutable v1 delivery `c951df0a660f43711ce25fbbe04591d1075299e87c2e345d92abe0ec074e419c`. No original/shared/live source, SDK, candidate packet, raw capture, `/dev/shm` path or ledger was modified. No proposed module was imported/executed, syntax-tested, built or GPU-run. Authoring scripts perform only CPU source/text operations.

## Exact scope

- **Alternate collector remains byte-identical0c02b36ae9942ccf5bf4792bd814ac13d1b589b564e1dae907b45128fb55f207** (v1's snapshot/preflight collector). It remains an additional `v100_roofline_collect_tmpfs.py`, never a replacement of canonical C21. No numerical/kernel source, math/parser, guard implementation, selection, profiler SDK/method or timing change.
- Storage **Policy remains root-reviewed native-v2a5fd1c62489557d266b59039ab272b898d86f57554b74fa0579c0bb552067efc**. This file is not changed here. Do not confuse it with the separate new pure **evidence validator**.
- `v100_tmpfs_evidence.py` gains `completed_checks`: frozen bytes/JSON/hashes only. No pathlib/open/SDK/tmpfs reads or execution of source snapshots. Existing whole original-plan inverse/native exact prefix-byte inverse/root-approved helper+validator+collector+guard/source joins remain unchanged.
- Summary supplies retained auxiliary manifest and execution log bytes to that validator. Its existing per-attempt catch/freeze/hold/continue logic remains unchanged.
- Audit requires the new **known exact validator SHA constant before exec** of retained bytes. The archive cannot choose its own validator by merely claiming a self-consistent provenance SHA. This gate is pinned in the root-adopted audit/frontier, independent of archive metadata. The later root-review/provenance/validator joins are still required.

Read the three nonempty `*.patch` diffs; collector diff is empty. `bases/` retains v1 exact candidate sources. These are private candidate tools, not adoption or permission.

## Completed-only source-contract gate

`profile_process_completed` must establish all of the following using retained bytes:

1. Process returncode0, no stop reason, no observed foreign GPU activity; full auxiliary retention `kept`.
2. Exact auxiliary-manifest SHA agrees with **both** top-level provenance and storage-summary hashes. Every inventory row has a unique canonical relative path, integer byte count and SHA encoding.
3. Exactly the five expected SU3 trial paths exist in that inventory,1,347,849,664B each, with SHA declarations. `large_artifact_validation` reports the same exact five absolute paths and actual/expected byte counts, all-complete=true. Original full-payload/per-file budget,256MiB auxiliary policy and zero fixture-copy policy remain unchanged.
4. Actual `execution.log` hashes to the completed process receipt. It contains exactly trials1..5, all `correct:true`, `jit_excluded:true`, original3internal repeats/region, seven finite nonnegative case times, full UInt32 count336,962,416/shape/little-endian schema and exact artifact path joins. Native also requires all seven original successful warm AoS markers. Julia's full warm check is source-bound and throws on failure; successful process plus original source and all-five checks remain distinct from a newly invented marker.
5. Actual guard source, per-review metadata, executed guarded argv and first-line Landlock attestation agree. The exact scratch leaf is allowed; broad ancestor or other `/dev/shm` allowances are rejected. Existing guard implementation remains unchanged.

**This is not a new full binary-value comparison.** The gate joins complete retained size/SHA/path inventory to source-defined actual scientific verdicts. It neither rereads volatile payloads nor certifies their future persistence from a hash. Root must separately archive retained full data before any authorized cleanup. Complete manifests/source logs do not imply sample phase, whole ROI, causal positioning or physics/application accuracy.

## Failed/held attempts

Preflight failure retains only available snapshots and returns no positions. Other noncompleted attempts may validate their available base source/storage evidence but skip all completed-only assumptions and receive a no-position qualification. Missing trials/partial failure-only diagnostics are not reconstructed, counted as checks or treated as zeros. Original status/process facts remain retained; summary's malformed/admission-error hold affects only that attempt and cannot invalidate a different qualified selected capture. Native's original256MiB auxiliary limit remains unchanged; possible partial diagnostic output is explicit.

## Adoption checklist — not performed

Root must independently source-review this complete packet, obtain a fresh review, adopt exact evidence-validator/summary/audit bytes and alternate collector while leaving canonical C21 and pending method followers pinned, and create a new source-bound CPU supervisor frontier. Update storage root review `adopted_collector_sha256=0c02…`, `helper_sha256=a5fd…`, `evidence_validator_sha256=NEW_VALIDATOR_SHA` and all fresh launch/read-only bindings consistently. Audit's source-pinned known validator hash already equals this packet's exact helper bytes; it is not a future wildcard.

Retained pre/post SDK/calibration admission, allreduce independent gates, generic derive math, primary scientific sources, default historical no-tmpfs behavior and strict12 source-counted-family baseline are unchanged. No actual new runtime validation, calibration or profiling outcome is claimed. No tests or imports were added/run. Shared preparation tokens are recorded separately from translation/compiler effort.
