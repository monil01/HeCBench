# Private flip-only merge identity exception

Root-current base: `root_merge_reviews_v5_callable_latin1.py`, SHA `a05280245cfd0297e459a34e107f460f50bc4df1ce1f773d570a210bd9f0c76a`. Candidate is source-authored only, not imported/executed/tested/applied. Exact unified diff is `minimal.patch`; no shared files changed.

Only when `configuration_id` comparison differs, it requires all of:

- Exact benchmark flip and one of the four model IDs; canonical CID remains null.
- Complete exact root scientific note `approved_contract_join_exceptions.flip` dictionary as shown in `root_note_required_field.example.json`. The note itself still must include actual packet hashes and all seven expected counts and is hash-bound in the approval plan/receipt as before. The example is **not approval** and is intentionally not a usable scientific note.
- Frozen raw config SHA `371c61a3cac140641d17e038c9118551a86626adc61201caf9d00b2c7f283a9c`; frozen result SHA `803e84790e80a9bc706967cb8633c542a83eeba0f00d29eff9ddf2d9a3477e7d`; these must also match both plan and canonical contract hashes.
- Frozen config explicitly says `flip_deterministic_n4096_r100_v1`, equal to plan, canonical review-record and proposal-record IDs. Frozen result must say benchmark flip and **lack configuration_id entirely**, not contain an explicit null or conflicting value.
- All unchanged original plan/provenance/config/result/stage/source/current selection/manifest/regex/whitelist/phase/ROI/causal checks still run. Ordinary configuration-ID mismatches remain fatal. All existing canonical and historical fields stay unchanged.

The receipt adds explicit per-model `configuration_id_join_exceptions` records with root-note/plan/provenance hashes, unchanged canonical null, absent historical result CID and narrowly descriptive identity-join qualification. The merge plan's resulting receipt hash covers these records, and required rootnote SHA already enters the plan. No inferred trial, source/ABI adjustment or acceptance relaxation is introduced.

Root may review this source, author a complete flip-only note/packet whose parent hashes are current, then run a fresh private dry-run and inspect the exact merge-plan SHA. Do not add flip to an applying batch or silently switch bound scripts. Root source-read and explicit approval are required before any later apply. No live source tools, accepted results, numerical criteria or ledger contract fields should be changed by this proposal.
