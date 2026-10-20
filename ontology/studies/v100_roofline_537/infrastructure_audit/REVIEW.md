# Independent profiling infrastructure audit

CPU/source/saved-evidence review only. No GPU, target/build execution, shared tooling/ledger/source edits, or foreign-process intervention. Snapshot and exact SHA bindings: `attempt_snapshot.json`, `source_bindings.json`. Files marked started are mutable observations, not final receipts. This is shared profiling effort, not translation/compiler effort.

## Current campaign progress

The scan contains 260 attempt records: 196 completed, 18 process failures, 38 pre-execution collection failures, 4 replay-method holds, and 4 started. All 38 collection failures report an existing foreign compute process on GPU6; retry only in fresh slots after the established idle check. No inference of framework failure follows. These are attempt counts, not distinct complete benchmark coverage.

Recent closed evidence includes backprop native (4 kernels, 2.077s target process), backprop CUDA.jl (11 kernels, 23.435s), attentionMergeState KA (4 kernels, 19.154s), and clock CUDA.jl (6 kernels, 16.016s). The controllers progress across serialized lock owners. Started provenance precedes lock acquisition, so its wall-clock age must not be interpreted as active GPU capture time. The actual `profile.started_at` and wall_seconds provide the executed-process interval. No deadlock evidence was found.

SLU is the clear expensive sampling exception. Native retained 412 geometries in 168.055s before the controller stopped **its own** profile on detection of external openmc PID1788160 at 20:27:10Z. CUDA.jl retained 777 geometries in 302.719s before its 300s operational deadline. Native partial rows: RL60, factor1, update350, clear1. CUDA.jl: RL60, factor1, update715, clear1. These are partial failed attempts and must not be promoted to completed coverage. Their 0.76MB/1.61MB CSV files do not show pathological disk growth; the cost is repeated profiling of many shapes with large mutable replay state. JACC at the final process snapshot was active CPU98.3%, had completed native CPU preprocessing, and had not yet emitted counter rows; this does not prove either a hang or eventual success. No intervention was made.

## Bounded SLU proposal (new profiling stratum only)

Use a fresh attempt and a reviewed selection override:

```
--filter-mode global --kernel-name-base mangled --kernel-id '::regex:.*:1'
```

Do not add `--kill`, reduce dimensions, alter inputs, change kernels, skip any of the five checked source trials, or replace the full numeric factorization with an isolated toy kernel. The source workload still executes in full; only counter selection is narrowed. Retain all old failures/counters. Source primary correctness remains its original LU/product/solve/backward-error contract, not exact floating atomic factor identity.

This is supported by the installed **NCU2022.4.1.0** help (saved full help/version): filter-mode default global applies filters collectively, per-launch-config applies separately to grid/block/shared-memory; kernel-id includes invocation number. Importantly, it is also backed by existing actual **shape_kernelid** canary (two functions, two shapes each, three repetitions): the global/default selector captured two rows, one per function; the combined per-launch-config selector captured four. Both exact CSV/provenance/source SHA bindings are saved. No new canary/GPU run was performed in this review. There is no assumption that this equals full geometry coverage.

Source mapping: `integrated_numeric.cuh` op0 RL, op1 RL_perturb, op2 RL_onecol_factorizeCurrentCol, op3 perturb factor, op4 RL_onecol_updateSubmat, op5 RL_onecol_cleartmpMem. Dispatch construction is at lines401–546. CUDA.jl/KA/JACC `slu_launch` maps the same op order to `(slu_rl!, slu_rl_perturb!, slu_factor!, slu_factor_perturb!, slu_update!, slu_clear!)`. `study_common.jl` wraps the nine original pointers including `10353*n` Float32 workspace; it requires all six authored entries to execute across the full control suite. KA `Val(smem/4)` and backend workgroup specializations may produce more than six mangled entries; **do not hardcode exactly six counter rows**. Preserve all observed specialization names. Distinguish control-case perturbation entries from ASIC default entries. First entry may be warmup or control state; retained-trial/representative-stage claims remain unproved.

A global first-specialization selection should remove the hundreds of update-grid samples without changing source computation, but its final duration and count remain unmeasured. Prefer this explicitly undercovered survey to blindly increasing the deadline on the current shape census. A full geometry survey needs a separate reserved long operational budget, or prospectively source-bound launch/stage/shape subsets with a disclosed enumeration and union; neither implies weighting or whole-factorization roofline aggregation. Root must stage native contention retry and CUDA.jl deadline retry in new slots. Do not interrupt active KA/JACC.

## Paths, libraries, and failed children

* Original ANS four source-relative diagnostics writes are observed Landlock denials; later copied wrappers isolate diagnostics. Native ANS copied-wrapper follow-up actually completed five source trials but failed its NumPy audit import. This is a user-space interpreter/dependency path issue; existing batch006 interpreter follow-up supplies the previously reviewed remedy. Do not rerun or silently accept the failed wrapper merely because counters exist.
* Original AMGmk four outputs write historical rawcapture roots. Those failures are covered by the existing output-only copied-common/native overlays; no new bypass or writable source-directory permission is justified.
* ASSERT Julia3 exit with `actual isolated assertion status mismatch` before a counter header. `run_study.py` checks BOTH the negative child return code and exact `[710]` status. The retained main log does not establish which condition failed. Fresh scratch diagnostics were manifest-only retained/removed by the collector; their hashes alone are not recoverable raw text. Keep this as failed instrumented execution, not parser malfunction or source failure. A separate profile-only positive-entry plan may be reviewed, with negative assertions excluded from profiling and their previously verified correctness retained as separate evidence. No automatic replay/scope authorization is supplied here.
* No observed SLU/backprop runtime-library loading or output-isolation failure: SLU fails by contention/deadline; backprop closes successfully. Loader paths, source hashes, exact selected plans and command provenance are available in the saved attempts. Do not change LD_LIBRARY_PATH or relax the write guard without a concrete loading error.

## Zero kernels and parser fields

There are **zero** completed-success attempts with kernel_count0 in this scan. Completed duration and DRAM-byte derived fields have zero missing values. The integer instruction counter is null for162 older sampled kernels: unknown/not collected, not zero and not a basis for an integer-only classification.

The numeric parser intentionally converts text metadata columns (device display name, function cache config, NVLink name/port lists) to null in `metrics`; all raw CSV bytes remain retained. These are not malformed arithmetic counters. Existing warnings that no counted add/mul/FMA/tensor was seen require source/SASS interpretation: integer/address/SFU/atomic/memory-only or eliminated work are possible. No blanket integer or no-work label follows.

Two narrowly scoped future hardening proposals (not applied):

1. Successful child + zero sampled rows -> explicit `profile_completed_no_profiled_kernels`; distinguish source-no-device-kernel not-applicable from filter/injection/child misses. Never synthesize zero FLOPs or completed roofline coverage.
2. For an explicitly requested numeric field failing conversion, record original token + field/unit/kernel identity + conversion reason alongside null. Preserve scientific verdicts and original CSV; never substitute0. Current parser rejects missing headers and duplicate metric identities, and flags unknown duration units with null duration rather than inventing values.

No new helper is required for the observed blockers; the actionable SLU fix is an explicit source-bound selector override owned by the parent, and the path fixes already have staged reviewed helpers.
