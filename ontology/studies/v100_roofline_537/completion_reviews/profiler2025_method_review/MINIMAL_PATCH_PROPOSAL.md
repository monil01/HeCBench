# Separate Nsight Compute2025.2 method stratum — private proposal

**Not adopted.** At this packet's read-only observation, the new calibration provenance says `calibration_prepared`, not pass. No missing/failed/pending result is approved. This packet changes no shared source, manifest, supervisor, measurement, primary timing or failure history. No GPU/build/tests were run.

## Existing gaps / exact code anchors

1. `v100_roofline_collect.py`, block immediately before `provenance['profile'] = run(...)`: version stdout, resolved `shutil.which('ncu')` and executable SHA are already recorded. Good basic identity, but command construction still uses bare `ncu`, version subprocess uses ambient environment rather than the exact child environment, and wrapper executable identity does not by itself identify the launched target ELF/injection libraries. Source calibration controller already snapshots these relevant bindings; reuse its explicit inventory rather than infer same tools from PATH.
2. Collector block `calibration = root/'ontology/.../v100_sm70_20261020/review.json'`: always loads the2022 calibration and checks only device UUID/architecture, then unconditionally passes half_lanes2. It does not join calibration tool identity to the actual profiler. Switching PATH alone can therefore create2025-derived numbers labeled by a2022 calibration.
3. `v100_roofline_classify.py::derive`: pure arithmetic uses supplied half evidence truthiness, factor and architecture/mode gates; it does not verify receipt files/version. Keep it pure and **unchanged** for this method extension: no numeric formula changes are needed. The campaign admission boundary must validate evidence before calling it, and audit must independently validate it. Standalone parser options are not an adoption certificate.
4. `v100_roofline_summarize.py::freeze`: fixed file allowlist omits a general method receipt/calibration archive; selection score/export currently lacks explicit profiler-method stratum. Raw provenance retains version but users cannot reliably filter selected CSVs by method.
5. `v100_roofline_audit.py`, just before recomputation of `derived.json`: reconstructs saved evidence parameters and exact frozen parser, which can reproduce a mislabeled calibration perfectly. Add an independent exact profiler↔calibration join for the new stratum, before accepting recomputed fields. Old historical parser and manifest bytes remain unchanged.
6. `v100_roofline_tensor_modes.py`: target source/API approvals bind the original2022 physical-mode evidence. That is necessary source-lowering evidence, but is **not** a2025 counter calibration. Retain that unchanged gate AND require a passed2025 metric convention receipt; never substitute a generic tensor presence check.

## Smallest root-applied implementation

Prefer one new CPU-only validator module, reused by collector/summarizer/audit, plus small hook changes below. This is a proposal for root implementation, not a silently applied patch. `adoption_contract_proposal.json` describes its required fields; placeholders are deliberately invalid.

Suggested validator API:

```python
validate_profiler_method(*,
    review_bytes, expected_review_sha256,
    observed_identity, device_uuid, architecture,
    retained_calibration_files,
    require_runtime_tool_recheck=False,
) -> dict   # validated method id + calibration receipt bindings; raises on any gap
```

Required checks:

- SHA-bound root-reviewed adoption receipt has exact status `root_approved_after_saved_calibration_pass`, method id and version scope. A canary pass by itself does not adopt the method.
- Exact full version stdout/build identity matches receipt; resolved requested wrapper path/SHA, actual launched target ELF path/SHA and explicit injection/runtime/section inventory agree with the canary manifest. Runtime collection checks these before/after; frozen historical audit checks retained original identity receipts rather than demanding today's installed binary remain present.
- Calibration source-binding manifest SHA matches; both `precision` and `arithmetic` cases finish successfully with returncode0, no stop reason, full host checks, source/ELF bindings unchanged, all9 exact named function/geometry identities, correct device UUID/rawPCI/sm70, all requested finite metrics and expected units.
- Recompute the frozen controller's pure `validate()` numerical comparisons from retained source-bound canary counters and exact old-reference comparators; equality of a status string is insufficient. Frozen controller/parser/metric inventory hashes must be audited. Expected scalar/thread counts and half/tensor physical counts are explicit in that function. No GPU replay is required for this audit.
- Missing, pending, failed, truncated, mismatched, ambiguous or unsupported calibration fails closed **before** a benchmark profile is executed/derived. If a benchmark process was already executed unapproved, preserve raw evidence but do not select its projections; do not relabel it retroactively as approved without a separate explicit review.

### Collector hooks

A. Add explicit optional `--profiler-executable`, `--profiler-method-review`, `--profiler-method-review-sha256` (propagate to collect). Resolve the chosen executable once; use that exact absolute command for identity probe and actual profile. Probe version with the same pinned child environment; record wrapper/target ELF and runtime inventory, then rehash after profile. Do not infer method merely from version substring or filename.

B. Before build/GPU execution, when requesting the2025 stratum, require the validated adoption receipt and full method bundle. An ambient2025 executable with no explicit receipt must be rejected, not assigned legacy2022 calibration. Keep the historical2022 route reproducible; future2022 commands should be explicitly labeled legacy2022 using existing receipt. Do not rewrite old manifests to add imagined identities.

C. For the2025 branch, replace hardcoded calibration selection with the validated new receipt. Keep this exact math:

```
THREAD pred-on counts: no ×32
FP32/FP64: add + mul + 2*fma
packed FP16 physical executed lanes: 2*(hadd + hmul + 2*hfma)
Volta WMMA HMMA.884 FP16input/FP32acc: tensor_inst*512
```

Scalarhalf may execute an unused packed lane: retain physical vs useful-work qualification. Keep architectural peak×observed clock ceilings distinct from empirically sustained performance. DRAM/L2 and scalar/tensor projections remain separate. No broader precision/architecture inference.

D. Store `profiler_method = {method_id, exact_version, executable_identity, adoption_review_sha256, calibration_manifest_sha256, calibration_provenance_sha256}` in provenance and `scope_context`. Pass the2025 receipt through `operation_calibration`, half evidence and tensor calibration linkage. Existing WMMA target typed approval still gates every tensor row; current-target SASS remains unobserved unless actually captured.

E. Write immutable slot files `profiler_method_review.json`, `profiler_method_calibration_manifest.json`, and a lossless source/counter/log calibration archive with original-path/hash inventory. No binaries or entire SDK must enter the paper repository. Record executable binary hashes; retain scientific source/counters/logs/controller/manifest/review bytes sufficient for CPU derivation audit.

### Summarizer hooks

- Extend `freeze()` allowlist only for new method receipt/archive files. Never alter old compressed profile bytes or calibrationreview.
- Validate the2025 receipt/bundle before qualifying a completed capture. A process returncode0 or reproducible math does not override missing calibration. Invalid method remains a retained failed/unapproved method attempt, not a positioning sample.
- Add method_id, exact version and calibration-review SHA to attempts, selected state and kernel/scalar/tensor exports. Missing legacy identity is `legacy_profiler_identity_incomplete`, not an inferred2025 label.
- Keep existing evidence-scope quality ranking. Prefer **per-method selected views** keyed by `(benchmark, model, configuration, method_id)`; preserve the existing legacy selected view until an explicit root policy chooses a new method. If a current one-selected-view policy is retained, label its selected method and expose both attempts; never numerically pool methods or allow lexical batch ordering to imply equivalence.

### Audit hooks

- For a new profile method id or observed2025 version, require frozen receipt/archive and validate its exact identity/scientific calibration join before deriving. No silently missing-method legacy fallback for2025.
- Recompute new canary scientific comparisons from retained bytes. Then execute the exact profile's frozen parser as before and require byte-equivalent derived values, preserving its method context. Do not run the calibration controller main or any hardware process.
- Preserve existing2022 historical replay branch unchanged; merely report its recorded legacy method/identity completeness separately. No new claimed2022 executable identity is added to old data.
- Emit separate counts by method id and explicit invalid/pending calibration diagnostics. Historical2022 numeric totals must remain identical.

## Root adoption sequence

1. Let the pending own canary finish. If unsuccessful, retain its logs/failure and stop adoption; no metric dropping or factor adjustment to obtain a pass.
2. Root independently reviews exact9-row science/units/PCI/SASS/source/tool bindings and full output checks. Freeze the new canary source, old/new counters, logs, manifests and controller/parser snapshots under a distinct new calibration directory; do not overwrite `v100_sm70_20261020`.
3. Create an exact immutable root-approved adoption receipt with real hashes and full identity. Pending observed provenance in this packet is **not** that receipt.
4. Pause/supersede manager/supervisor invocations whose source hashes bind old tools. Root applies/reviews the small hooks and new validator, records new hashes and starts a fresh separately named batch. No in-flight bound process silently changes method.
5. Collect no new benchmarks under2025 until the gate accepts the actual chosen executable+runtime; keep all original GPU6 authorization/UUID/PID/writeguard/state/replay/5check rules.
6. Refresh/audit/export under metadata lock. New method fields remain filterable; plot separate facets/symbols if both methods appear. Durations, traffic, clocks and cache effects need not equal2022 calibration values; no performance equivalence or causal tool-fix conclusion follows from matching instruction conventions.

## Expected acceptance / rejection outcomes (specification only, not executed tests)

| Situation | Expected behavior |
|---|---|
| Pending status calibration_prepared | reject2025 adoption/derivation |
| Final calibration failed, missing host check, missing metric/row or changed binary | reject; preserve raw failure |
| Passed9-row canary but no root-approved receipt | retain pass evidence; still reject campaign adoption |
| Receipt exact2025 version but command resolves other wrapper/targetELF | reject before profile |
| Exact method receipt but targettensor mode not WMMA-approved | half/scalar permitted if otherwise eligible; tensor conversion remains null/unknown |
| Exact2025 receipt and targetWMMA typed approval | conditional512 factor only; method id new, scope unchanged |
| Frozen historical2022 profile | exact old parser recomputation unchanged; no manifest edits |
| New2025 rows with no method archive | audit hard error, not silently admitted |
| Counter/profile execution failed | no positioning from fragments, regardless of calibrated tool |

No tests were added/run. Input tool source snapshots and hashes are included so root can rebase the proposal if current files change before adoption.

### Calibration coverage qualifications to retain

The pending controller's9rows cover packed/scalar half FMA, HADD2/HMUL2, WMMA and FP32/FP64 FMA specializations in two geometries. Do not describe this as independent fresh calibration of every scalar add/multiply opcode or every tensor precision. FP32/FP64 weighting1/1/2 remains the metric's scalar instruction semantics; the canary checks its actual typed FMA counts. Existing canary binaries are reused and hash-bound; their old compiler identity was not captured, so no newly observed original compiler version/source-to-binary rebuild claim follows. Selected target instruction lowering still needs its own source/SASS or API evidence.
