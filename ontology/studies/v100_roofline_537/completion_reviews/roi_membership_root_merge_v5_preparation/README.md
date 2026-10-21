# Private v5 merge-engine proposal

Base: frozen `roi_membership_root_merge_v4_preparation/merge_reviews_v4.py`, SHA `ee3068f93f97b2a618b2c5aafe08e39056b5eb234e8705d0a597f5545360b5ee`.

This packet only authors source and a unified diff. **The candidate was not imported, executed, tested, dry-run or applied.** Root must review and perform actual dry-run validation before approving its exact resulting merge-plan SHA for apply. Source/selected state might change before that validation; no counts are claimed validated by this preparation.

Changes:

1. Required `--root-scientific-review PATH`. Its exact resolved path and bytes/SHA enter the existing plan/receipt; all packet SHA bindings and final review-byte stability remain enforced.
2. Expected counts come from that note's `expected_counts`: exactly the original seven key names, nonnegative true integers, complete dictionary equality with actual derived counts. No fixed 56/14/124 population remains. The currently supplied batch4 note declares 80 model records/20 names/146 inclusion declarations/88 exclusions/0 pure alias nulls/5 inclusion alias nulls/141 non-null inclusion stages, but only root's actual validation may certify them.
3. New `roi-v5-*` transaction namespace and `.roi-v5-active-commit.json`; **either** unfinished v4 or v5 marker blocks validation/apply. Existing EX/SH locking, all-parent validation, whitelist, frozen selected joins, history archives, two ordinary directory renames, fsync/journal, exact rollback and fail-closed crash recovery are preserved. This remains explicitly non-atomic and requires all readers to honor metadata locks.
4. Only clock Julia3 may use `capture_overlay_provenance.patches` retained-source evidence. It verifies exact selected compressed-plan binding/pointer, whole retained text SHA/copy SHA, original/copy pins in captured execution bindings, present original SHA and exact historical scratch directory/entry identity. Rules are built independently: common rawdir becomes `STUDY_OUTPUT_DIR/private_clock`; its one fixture anchor becomes the canonical clock source directory; model entry includes the same historical copied common. The common is also independently validated when only entry evidence is provided. No arbitrary changes array/affirmative booleans are trusted. Exact forward and inverse bytes are checked; no absent historical file is recreated. New clock checks are separate from retained AMG checks; existing AMG logic is unchanged.

## Root-only next commands

Use actual fresh private output paths selected by root, not existing 001/002 output trees. From this folder:

```
python3 merge_reviews_v5.py \
  --source-root /home/imo/HeCBench_julia \
  --output-root /tmp/ROOT_CHOSEN_FRESH_DRYRUN \
  --root-scientific-review /tmp/roofline-completion-20261021/roi_batch4_root_scientific_review.json \
  /tmp/roofline-completion-20261021/roi_membership_source_batch4/proposed_stage_patches.json \
  /tmp/roofline-completion-20261021/roi_membership_batch4_stencil/proposed_stage_patches.json
```

The displayed output path is illustrative, **not a ready command or reserved directory**. Preserve this exact packet order when root later applies with `--approval-plan` and `--approval-plan-sha256`; full regenerated plan equality still enforces order. Root must inspect and approve the dry-run report, seven counts, source/retained-overlay checks and unchanged scientific phase/fullROI/causal facts, then use a different fresh private output directory for explicit `--apply`. No acceptance widening, sample-phase promotion or numerical/performance changes are authorized by the source proposal.
