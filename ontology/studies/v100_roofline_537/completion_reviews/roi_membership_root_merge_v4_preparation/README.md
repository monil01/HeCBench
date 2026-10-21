# V4 explicit lock-serialized journaled two-renames

This is a NEW commit method, never a silent fallback. The previous v3 source, approvals, root apply001 argument-order failure, and root apply002 renameat2 EINVAL remain immutable. No canonical adoption is implied by their preparation/history side effects. This v4 default read-only validation passed; --apply has NOT been executed by the worker. No GPU/build/tests or shared metadata changes were made.

## Exact root-only command

```
python3 /tmp/roofline-completion-20261021/roi_membership_root_merge_v4_preparation/merge_reviews_v4.py \
 --source-root /home/imo/HeCBench_julia \
 --output-root /tmp/roofline-completion-20261021/roi_membership_root_merge_v4_applied001 \
 --apply \
 --approval-plan /tmp/roofline-completion-20261021/roi_membership_root_merge_v4_preparation/validation_001/merge_plan.json \
 --approval-plan-sha256 417ab3d33c550fea1a86669b9a07b128b8422a85e6f29cdc610b770ef6ff81bf \
 /tmp/roofline-completion-20261021/roi_membership_science_infra_v3/proposed_stage_patches.json \
 /tmp/roofline-completion-20261021/roi_membership_science_stencil_v3/proposed_stage_patches.json \
 /tmp/roofline-completion-20261021/roi_membership_science_cold_v3/proposed_stage_patches.json
```

Packet order is part of the approval digest: INFRA, STENCIL, COLD. Root must inspect/freeze this new engine before invoking it. Readers MUST hold the existing metadata SH lock, writers EX; a noncooperating reader could observe the brief absent-directory interval. This is NOT an atomic filesystem exchange or a generic multi-file transaction.

## Preflight and commit

The approved plan includes actual /proc/self/mountinfo filesystem type/mountpoint/options and st_dev equality. The dry run does not test rename support; actual root ordinary-renames remain an operational prerequisite. Target, parent and fresh staged tree must be on the same filesystem; all review children are regular files; history directory cannot be a symlink. Exact initial gzip/INDEX/record/stage/source/capture checks remain those of reviewed v3. Fresh transaction paths cannot collide with v3 staged paths.

Apply validates everything under metadata EX, stages the whole new directory and in-directory receipt, archives exact changed old bytes, then creates and fsyncs an in-repository .roi-v4-active-commit.json marker. This contains exact old/new tree hashes, target/stage/backup paths, approval digest and transaction id. Both marker and private transaction_journal.json are updated before/after each rename. Ordinary same-filesystem target->backup then stage->target renames run under EX with parent fsync. Only after complete new canonical and old rollback tree verification is the marker removed. Old rollback tree is retained as a hidden sibling; do not accidentally add it to Git.

## Failure and crash recovery

Catchable errors inspect actual full tree bytes. If target is exact new and backup exact old, target moves back to stage, then backup restores target. If target is absent and backup exact old, backup restores target. Already unchanged original is accepted only with no conflicting backup. Any unknown/missing/conflicting bytes leave the marker and fail closed; no overwrite guessing occurs.

Power loss/kill can leave the canonical path absent between renames. New invocations refuse any existing marker, including a symlink marker; they do NOT auto-recover it or start another transaction. Root recovery must acquire metadata EX and inspect actual full trees against marker/approved-plan hashes, not trust journal phase alone: (1) target exact-old means no adoption; (2) target absent + backup exact-old permits deliberate restore; (3) target exact-new + backup exact-old permits explicit retained adoption or deliberate rollback; (4) anything else is unknown and requires investigation. Preserve marker/private journal/stage/backup before action. Never clear marker merely because a PID disappeared. No worker recovery/apply was attempted.

Validation counts remain 56 models, 14 names, 124 inclusions (2 alias-null inclusions, hence 122 non-null stages), 13 exclusions and 25 pure alias nulls. No actual sample-phase, complete-ROI, multiplicity or causal promotion. Later summary/integrity audit remains mandatory and separate; strict12 integrated scientific family admission is unchanged.
