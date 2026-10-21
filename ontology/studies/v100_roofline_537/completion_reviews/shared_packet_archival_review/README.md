# Deterministic static archival proposal — not applied

Four CLOSED shared windows and linked private v5 method candidate. archive_manifest.json inventories94 static files:25,429,646 original bytes,4,380,817 archive bytes. Files>256KiB gzip-compressed with mtime0 and recorded original/compressed SHA+length; all other bytes preserved. This matches the study's raw+compressed provenance convention while retaining plain JSON receipt paths required by SHARED_EFFORT_INDEX schema1.

Exactly four index entries proposed, same existing allocation `shared_nonexclusive_not_per_benchmark_translation`. Existing receipts rehashed under LOCK_SH before preparing; new boundary event pairs checked for duplicates and partial overlaps against existing index and one another. Root window remains OPEN and unchanged, no sum_prohibited_reason change, no token total added. Updated_at derives deterministically from actual latest recorded receipt boundary/index time. Different receipts may overlap temporally: indexing does not authorize adding them.

The calibration003 receipt expressly covers actual002 unit diagnostics,003 source/controller/manifest, AND private v5 preparation. Its frozenv5_DELIVERY_sha256 exactly joins v5/DELIVERY.json. v5 has NO additional receipt or token allocation. receipt_joins includes an explicit link to that one003 receipt. This does not assert calibration003 pass, live method adoption, or cold independent per-subtask effort.

Active future_controller_receipt.json/future_follower_receipt.json and future_*.log/jsonl, lock/temp/new files or cache files are excluded by source rule. None occur inside these five closed source packet trees at this snapshot. Private future command/proposal source is static evidence, not executed. Frozen historical logs inside preserved002 are retained, not mistaken for mutable future logs. Symlinks are forbidden. Every archived file retains its original path and exact bytes/hash; archived code is evidence only, never automatically adopted or launched. No raw agent transcripts are included.

## Root review/application

`apply_archive.py` has NOT been imported/run/tested. Read it first. Default is no repository writes; --apply is root coordinator's explicit archive/index mutation step. It validates exact manifest SHA, payload hashes/decompression, safe destination prefix and absence of symlink paths, rechecks all static sources, then obtains existing LOCK_EX and requires exact old index SHA before copying. Existing identical archive bytes are reusable; any differing file is preserved with rejection. Index update is last and atomic. An index change requires a fresh rebased/reviewed proposal rather than overwriting concurrent receipts. No source/Git mutations, GPU calls, build/test or candidate execution.

Review command template (NOT run):

    python3 apply_archive.py --source-root /home/imo/HeCBench_julia --proposal-root /tmp/roofline-completion-20261021/shared_packet_archival_review --manifest-sha256 MANIFEST_SHA

Root may add --apply only after its review. The archive destination is ontology/studies/v100_roofline_537/completion_reviews, with original packet hierarchy. proposed_SHARED_EFFORT_INDEX.json is a proposal, not a replacement for a newer index. No archived packet command should be rerun as part of archiving.

This worker's archival-review receipt is separate and not one of the four requested additions; packaging after END excluded. No campaign sum or translation attribution.
