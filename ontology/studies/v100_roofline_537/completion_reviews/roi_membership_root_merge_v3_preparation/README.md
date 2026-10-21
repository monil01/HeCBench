# Root-only source-ROI merge proposal v3

## Validated scope

Default CPU read-only validation completed under the existing metadata LOCK_SH; no GPU, builds, tests, or shared writes. All initial gzip parents, INDEX hashes, canonical record/stage hashes, selected frozen plan/provenance/derivation/config/result joins, and source snippets were checked before admission. The root scientific review is an exact input binding. Whole source bytes were checked before/after validation. This is source membership annotation, not new numerical verification.

Validated: 56 models / 14 names; 124 inclusion declarations, 13 exclusions, 25 pure alias-null changes. Two inclusion declarations are themselves shared-family aliases (BM3D), leaving 122 non-null inclusion stages. Adding 124 to the previous 118 gives 242 source callsite declarations, not 242 independent measured families. No actual-sample phase, retained-trial identification, launch multiplicity, full-ROI, or causal conclusion is promoted. Existing root-adopted 25 names remain unmodified.

## Exact quote handling

The failed private validation_001 and validation_002 reports are retained. They exposed validator schema handling, not changed study sources: trailing blank context lines and quoted contexts capped mid-long-line. The final validator checks ALL supplied context characters exactly against the source beginning at the declared line; it does not pretend a capped quote is the complete underlying line. Explicit byte intervals and line quotes are checked separately, with byte-based newline counts (including Unicode source). No packet evidence was changed or truncated further. AMG historical overlay files are absent; retained full text in the exact selected immutable plan hashes to the actual executed source, and only its documented path substitutions are reproduced IN MEMORY. No old file is recreated.

## Root adoption command

Use the exact frozen script and approval plan below. Default validation has already run; --apply has NOT been run. Root coordinator only should execute the following after review, with a fresh output directory:

```
python3 /tmp/roofline-completion-20261021/roi_membership_root_merge_v3_preparation/merge_reviews.py \
 --source-root /home/imo/HeCBench_julia \
 --output-root /tmp/roofline-completion-20261021/roi_membership_root_merge_v3_preparation/root_apply_001 \
 --apply \
 --approval-plan /tmp/roofline-completion-20261021/roi_membership_root_merge_v3_preparation/validation_003/merge_plan.json \
 --approval-plan-sha256 40107c3adee4d0a69b3e9aa9639e90e71df4e8fc3bae616de0720e3b63948abc \
 /tmp/roofline-completion-20261021/roi_membership_science_infra_v3/proposed_stage_patches.json \
 /tmp/roofline-completion-20261021/roi_membership_science_stencil_v3/proposed_stage_patches.json \
 /tmp/roofline-completion-20261021/roi_membership_science_cold_v3/proposed_stage_patches.json
```

## Transaction and recovery boundaries

Apply revalidates the identical approval plan under metadata LOCK_EX, stages the complete regular-file directory as a same-filesystem sibling, archives every changed old file/INDEX by exact content hash, and commits the complete new roi_reviews directory plus INDEX and in-directory enrichment_transaction.json using Linux renameat2(RENAME_EXCHANGE). There is NO sequential file-replacement fallback. All target children must be regular files; symlink children/subdirectories are rejected. Root should confirm the existing roi_review_history directory is the normal intended archive directory before apply. The canonical directory path remains constant, but its inode changes; readers must honor the metadata lock and should not hold stale directory handles across transactions.

The hidden sibling retains the entire prior directory as a rollback tree. Preserve it outside Git staging. Catchable post-exchange errors trigger a whole-directory reverse exchange under the same lock. A process crash/power failure is not claimed automatically recoverable: inspect private transaction_journal.json, canonical and rollback tree hashes, and the authoritative in-directory receipt before acting. Atomicity guarantees an entire old or new tree, not a transaction across later summaries/audits/paper files. Do not rerun apply after an uncertain outcome without this inspection. Unsupported renameat2/filesystem fails closed.

After successful root apply, run the normal source annotation summary and full integrity audit under the existing coordination. The merge receipt explicitly remains pending that audit. No manager-bound tools are changed, and no observed sample phase or scientific-completion claims are strengthened.
