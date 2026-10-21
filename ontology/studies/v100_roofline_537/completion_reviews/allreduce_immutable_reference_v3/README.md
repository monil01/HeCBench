# Private allreduce proposal v3: immutable source reference only

**Not staged, executed, imported, built, tested, adopted or authorized.** GPU5
permission remains pending. No shared source/canonical review/authorization changed.

## Exact change

`proposal_v3.json` replaces the reference-only live path
`/home/imo/HeCBench_julia/tools/v100_roofline_collect.py` with
`immutable_reference/v100_roofline_collect_32a9cfeba983.py` in seven occurrences:
four model execution-binding map keys, the global read-only-binding map key, the
prospective calibration-binding map key and counter_contract.collector_path.
The reference bytes and SHA remain exactly32a9cfeba983dc3ae8e5d9b8bb85ca4e7736821be0a9c206bb57b80c7471da4c.
The new private file has read-only0444 permissions; no source code is edited.
Revision metadata changes2→3. Every other proposal value is identical after the
inverse one-path substitution. The exact JSON unified diff is retained.

The original v2 proposal4a950529... and runnerb7563cba... remain untouched, with
exact snapshot copies in this packet. Source-equivalence JSON proves all four
whole model objects have byte-identical serialized forms after inverse substitution.
Commands, source SHA values, target/header/rank-helper paths,32requestedmetrics,
required units/device checks, five correctness trials, geometry, profiler ELF,
observer/queue/guard policy, and two-device authorization requirements are unchanged.
All71 original read-only input bytes were checked against their existing hashes
at preparation. This is a sequential CPU source/data check, not runtime validation.

## Why it is a reference, not a target

The actual custom runner only verifies the global map and freezes its .py source
bytes into source_snapshots.json. Its collector_source.py is the unchanged actual
custom runner. Its only dynamically executed scientific support is the exact
counter parser and checker, not v100_roofline_collect.py. Build commands and
rank dispatcher/worker target commands are unchanged; the old live collector path
appears in no executed command. The v2 source-equivalence preparation script stays
hash-bound but is not invoked by the runner. Its historical references are not a
remaining live dependency of execution.

Future invocations must use the **original v2 runner at its original path** with
`--proposal /tmp/roofline-completion-20261021/allreduce_immutable_reference_v3/proposal_v3.json`.
Do not relocate or execute the runner snapshot: HERE remains the original v2
helper/checker directory. `root_launch_template.json` is nonexecuting metadata,
with mandatory root/authorization/output placeholders, not a runnable launcher.

## Root review and permission remain separate

The prospective CPU-review template changes its proposal path/SHA and collector
reference to v3/frozen32a9. It retains root_reviewed_for_cpu_adaptation=false;
it is not written over the canonical root review. Root must freeze a fresh actual
CPU-review identity, and if the user authorizes GPU5, record a fresh authorization
receipt with new proposalSHA, unchanged v2runnerSHA and the new CPU-reviewSHA.
Existing v2 authorizations/reviewSHAs must not be silently reused. No permission
record or actual capture is created here.

The classifier, Landlock guard and all old2022 calibration supporting paths remain
bound and unchanged. This only decouples allreduce from future edits to the unused
live collector reference. Calibration002 itself still pins the live32a9 collector:
this packet does not permit changing that file before calibration002 completes,
nor does it establish any2025 profiler calibration/adoption result.
