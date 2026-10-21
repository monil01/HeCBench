# Private restricted allreduce ingestion proposal

Source preparation only: no candidate helper/adapter/summarizer/audit imports or
execution, no GPU/build/tests, no shared application. Earlier adapter5562683 and
all v1/v2 runner packets remain unchanged. This packet adds **new** adapter_v2.py,
new admission helper and unified diffs against retained exact summarizer/audit
preimages. The generic `read_csv`/`derive` code and generic audit recomputation
block are unchanged.

## Concrete admission and hold behavior

The helper is restricted to method
`allreduce_two_device_rank1_private_add_NCU2022_4_1_v1`, benchmarkallreduce, and
four canonical models. Its authority is the root-owned canonical JSON:
`ontology/studies/v100_roofline_537/completion_reviews/allreduce_two_device/root_cpu_review.json`.
A self-approved flag embedded in an arbitrary capture cannot replace that file.
Exact root-review, helper, frozen adapter, actual custom runner, parser, envelope,
parent plan/provenance, selected result/config, authorization, raw inventories,
source snapshots and counter bindings must join.

The helper calls the exact frozen adapter's `validate_slot` on the **actual**
original private capture tree, with the exact frozen parser. It hashes every
original parent file before and after, independently checks every full value in
all5 full5,269,603-word captures and both saved GPU-reference artifacts, both-rank
process/UUID/PCI attestations, original guards/96-dispatch/five-trial markers, all32
required counters, old2022 calibration and complete scoped geometry. It verifies
retained compact raw archives against actual original files; unavailable raw
parents or artifacts cannot pass by presenting hashes. Root permission is checked
from the exact retained authorization bytes and captured receipt SHA. No original
source identity is reconstructed from current source-file hashes.

Root-approved source loading is explicit: only after canonical authority/helper/
adapter/parser SHAs match do audited frozen code bytes load. No process or GPU is
launched. This CPU-only validation can be expensive (full parent/artifact reads
and independent Python checks), so it must not be mistaken for a lightweight hash
certificate or a new performance/translation attempt. No validation cache added.

Summarizer catches method-validation failure **per attempt** and freezes available
bytes without aborting unrelated benchmarks. It sets
`profile_completed_allreduce_method_validation_hold`, suppresses that attempt's
exported points/observations, and records the exact error and original process
status. Historical raw rows/derived files stay intact. An earlier better valid
selection still wins by the existing policy; latest hold is separate.

Audit revalidates independently. A selected/completed allreduce failing admission
is an integrity issue. Unselected held evidence may still pass frozen-byte and
generic mathematical integrity but is explicitly labeled
`pass_frozen_integrity_method_validation_hold`, with qualification and
`positions_publishable:false`; that is not an admitted method or a fresh artifact
pass. This distinction permits zero corrupted frozen files without concealing
remaining measurement holds.

## Write-guard compatibility

Current collector uses `{tool,sha256,write_directories,policy}`. New adapter emits
that shape, with `write_directories` equal to the **actual original** fresh slot
used by the custom runner, not the CPU-derived output directory. It decodes the
exact pre-target Landlock source from the actual source snapshot into
`write_guard_source.py`. Admission verifies its SHA and all original build/profile
argv have exactly `PYTHON GUARD --allow-dir ORIGINAL_SLOT -- ...`. No unrestricted
fallback, source-parent write grant, or new writable directory is introduced.

Current audit32a9-era source does not itself contain a generic guard assertion;
the new helper explicitly enforces this method's guard/source shape regardless.
The existing freeze list already retains write_guard_source.py. New freeze names:
`adapter_source.py`, `allreduce_two_device_review.json`,
`allreduce_cpu_review.json`, `allreduce_admission_source.py`,
`allreduce_execution_authorization.json`.

## Method labels and process accounting

CSV model/kernel/DRAM/L2/tensor records gain `measurement_method`. This method's
ROI coverage is `single_source_marked_rank1_private_add_not_collective_ROI`, not
generic first-warmup or a whole collective. Source runtime checks prove one add in
trial2/n1M/firstrepeat, rank1 physicalGPU6, while both ranks still execute all
original checks. It cannot imply measured MPI/NCCL/communication cost or rank0,
other-size, dispatch-multiplicity or wholeROI extrapolation. Causal bottleneck stays
unknown and old primary mean2–5 remains unchanged.

The model CSV includes `process_unit_accounting`: this CPU adaptation creates0
GPU/build processes, while the original parent has1 monitored profiler/MPI
launcher and the actual number of build launchers (native1; Julia typically2,
but helper uses actual receipts). Underlying rank/compiler/device-JIT totals remain
unknown. If parent and adaptation are both separately frozen, census must deduplicate
execution totals by original-parent provenance SHA while retaining both evidence
records. No census code is changed here; until its explicit method support is
adopted, the CSV qualifier is mandatory and no combined process-total claim is
supported. Raw original process dictionaries are preserved exactly.

## Root adoption order; no permission inferred

1. Review private sources/diffs and exact active preimages. Copy the new helper to
   tools/v100_roofline_allreduce_admission.py and adopt summarizer/audit diffs only
   after superseding hash-bound manager tool versions. This worker applied nothing.
2. Construct canonical CPU-review JSON from the deliberately unapproved template.
   Set exact final helper/adapter/proposal/runner/parser/calibration hashes; choose
   a fresh **nonbatch** private parent `capture_root` so the ordinary scanner cannot
   see incompatible custom casefiles. Choose future user_authorization_path.
   `root_reviewed_for_cpu_adaptation:true` authorizes only the CPU evidence method.
3. Ask for explicit GPU5 permission as the final execution approval. If granted,
   root records exact runner authorization with **additional**
   `cpu_capture_adapter_review_sha256` equal to the already-frozen CPU-review SHA.
   The existing runner accepts additional receipt keys. Keep the root review's
   future authorization path but **not its SHA** in that review: this avoids a
   circular review→authorization→review hash dependency. Authorization must retain
   explicit user reference and proposal/runner/device/model/deadline fields.
4. Only root may run the separately reviewed cold-agent v2 runner after permission.
   No capture exists or is claimed here. Root CPU review, idle5/6 or old accepted
   two-device numerical results do not grant that permission themselves.
5. After terminal captures, run adapter_v2.py with exact canonical-review path/SHA,
   original private capture root and a fresh private output directory. Root can
   promote only complete derived batch evidence after review; do not reserve or
   write051–053 (already owned), and do not expose original custom casefiles under
   batch* to the ordinary scanner.
6. Refresh with the reviewed summarizer, then audit with original parent trees and
   complete artifacts still available. Failures remain holds; no wholeROI/causal
   adoption or new scientific correctness claim is automatic.

## Pending canary / active source bindings

The83-file NCU2025 calibration manifest binds collector32a9..., parser3878f43...,
Landlock guarda200738..., old calibration and canary/profiler files. It does **not**
bind summarize.py, audit.py, failure exporter or this new helper. This packet
changes none of those83 bound files. Root's finite manager separately binds
summarize/audit, so a new manager/source adoption is still required. No active
controller or source was modified or stopped.
