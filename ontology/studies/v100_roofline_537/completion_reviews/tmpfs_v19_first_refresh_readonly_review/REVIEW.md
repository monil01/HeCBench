# First actual v19 refresh: independent read-only review

The existing CPU refresh1662499:start78558281 completed its first saved cycle successfully through joint_checkpoint at2026-10-22T00:35:20.092732Z. All saved stages returned0: summary, integrity audit, plot, ontology report, publication snapshot, retained-family aggregation, benchmark assessment and joint checkpoint. The reviewer did not execute these tools; it read their actual output and frozen/current files.

## Exact default/scientific baseline preservation

Strict-family audit `/home/imo/HeCBench_julia/ontology/studies/v100_roofline_537/retained_roi_aggregates/audit.json` SHA remains **fbab226ce4adfb94cef62acdb9479266bf5693e9b184b6023e6a74db1192263d**, exactly matching the immutable `roi_batch7_strict_family_baseline_after_method_schema.json` baseline. This is whole-file byte identity by SHA, including all scientific records, method metadata and saved input hashes, rather than a normalized-field comparison. Eligible families remain12 (triad/BabelSTREAM/Jacobi×4), elapsed-host-ROI positions0 and whole-benchmark causal claims0.

All three current ledger byte hashes still match those frozen in the strict audit:

- triad:2ca181b07495d29f6f69db61d88d024a10ea97eecd2aa43d3af7bf1cd70cc013
- babelstream:b6d93587f8a9f1a5e2a61963ac1edca9f154886229ce1c89f10a7a163cf0904d
- jacobi:18539fdad7efc01d5f54602df8582efeb96b0e27c3625b0340657ea79c0ecf8e

Actual integrity output reports1808 retained attempts,18414 recomputed kernels,4parser versions and0issues (480qualifications). These are saved CPU-audit results, not new GPU trials, a complete-ROI certificate or a newly inferred numerical claim.

## Deployed source identities and queued063

Canonical collector C21c21b1edd…, parser3878f43f… and guard a2007386… match their expected full hashes. Additional collector0c02b36a…, Policy a5fd1c62…, pure evidence validator8dcbeed4…, summary7347d38d… and audit9dd23dbf… also match the reviewed exact source hashes. Audit requires known8dc validatorSHA at line85 before retained-code exec at line88. No source was changed by the reviewer.

Actual supervisor1662498:start78558271 and follower1662497:start78558240 retain exact expected identities and no children in the final read-only snapshot. Follower receipt says waiting_exact_predecessors, collector_invocations0, GPU_queries0 and signals0. Whole predecessors1160651 and1283663 remain live;1193331's historical generation is absent. Neither the reviewed `/dev/shm/hecbench-v100-uid10384-su3_batch063_root_review_pending` namespace nor raw063 directory exists (also checked with lexists, not merely directory existence). This is queued work, not a capture/admission result. No GPU query was performed by this review.

## Informational restart metadata discrepancy

The newly written CHECKPOINT.json still copied `roofline_campaign.active_work.cpu_refresh_supervisor=1471189` from STATE/active_work_537.json, while the actual current refresh is1662499. Historicalv14–v17 nested notes also retain1471189. The historical notes should remain historical; the current active fields/v19 links should be updated by root. Root was notified. This is separate from the scientific/default-byte preservation result; no ledger/tool mutation was made by the reviewer. `restart_note_observation.json` binds the exact observed checkpoint/state hashes, so a later root correction does not invalidate the historical observation.

`actual_saved_snapshot.json` and `refresh_stdout_snapshot.jsonl` retain exact actual observations. Metadata comparisons used existing LOCK_SH; process/storage/source observations are sequential rather than a simultaneous system snapshot. No extra exporters/auditors, benchmark tests/builds, GPU queries/runs, signals, proposed imports or shared writes occurred. Shared review usage is retained separately and is not translation/compiler cost.
