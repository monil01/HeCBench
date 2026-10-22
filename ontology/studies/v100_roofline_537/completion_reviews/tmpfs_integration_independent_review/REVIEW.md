# Independent tmpfs frozen-evidence admission review

**Hold deployment pending two corrections.** Exact candidate delivery `c951df0a660f43711ce25fbbe04591d1075299e87c2e345d92abe0ec074e419c` and all15 listed file hashes/sizes match. Review is source-only: no candidate imports/execution, tests, builds, GPU queries, live/shared edits or tmpfs writes. The packet remains untouched.

## Blocking: incomplete successful-storage tuple

`proposed/tools/v100_tmpfs_evidence.py:validate` never receives or checks `profiling_auxiliary_manifest.json`, despite the collector retaining its SHA in both `prov.auxiliary_manifest_sha256` and `prov.reviewed_tmpfs_storage.manifest_sha256`. Summary's available-file list excludes that manifest. Thus the validator's successful return does not establish its mandatory full-payload inventory, even if these fields are missing/inconsistent. It likewise does not require successful-process rc0/stopNone/no foreign activity, complete `large_artifact_validation`, or all five actual correct STUDY_TRIAL rows in execution.log before returning positions_publishable:true. The current SU3 native source reports per-trial correct:false without necessarily changing its final process exit; rc0 alone is therefore insufficient. No actual failed/new tuple was executed to demonstrate this; it follows directly from the absent checks and source control flow.

Required narrow fix: for completed status, consume frozen auxiliary manifest and execution log bytes; join manifest SHA to both provenance fields; validate exactly the five required SU3 paths/sizes/nonemptySHA against the source-declared large-artifact rows and actual five correct log trials, full dtype/count and timing/scientific contract. Require rc0, stopNone and empty observed foreign activity. Keep this an **offline manifest/log/source-contract check**, not invented full raw-value re-verification from hashes or a live volatile-file lookup. Retained failed/preflight attempts remain no-position qualifications with explicitly partial evidence.

## Blocking: executing an untrusted validator before its admission

The proposed audit joins frozen validator SHA only to its provenance link, then execs those bytes. The root-review validatorSHA comparison occurs inside that already-executing function. It must precede execution. Summary already compares retained validator bytes with the known adopted module's bytes before calling validate; audit should make the equivalent trusted-source-byte gate first (or an independently pinned known SHA), then execute/use the approved exact bytes. An archive's self-consistent source/provenance hash pair is not sufficient source admission.

## Checks that pass / preserved scope

The validator itself is pure byte/JSON/hash processing: no Path/open/subprocess or live SDK/tmpfs reads. Full original/fresh-plan inverse is checked, including exactly one native execution-bindingSHA normalization and full native source-byte forward/inverse prefix proof. Snapshot inventory rows enforce retained SHA/size; base snapshots and both native snapshots are required on the successful path. Root review joins helper, collector, guard, validator, device, selected config/result and plan source bindings. Failed preflight has a distinct partial-evidence/no-position result.

Summary's new storage exceptions are caught per attempt and continue after retaining the failed packet. They do not invalidate a different selected capture. Existing no-tmpfs paths and generic derive/parser arithmetic are unchanged. Canonical C21, SDK methods, allreduce admission, original SU3 scientific sources, primary timings and strict12 source-counted family authority are unchanged. Nothing in this proposal can promote actual sample phase, whole ROI or causal status.

## Follow-up authority

Root agreed both blockers and requested a separate private v2 patch in a new prospective shared phase. Original author stencil_ports acknowledged no overlapping edits and preservation of c951df0a. This review grants no runtime/deployment/admission permission. Root must review the new exact sources and frontier before adoption.
