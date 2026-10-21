# Private method patch v4 — root staging only

This new proposal addresses the two independent-review findings. V3 and its review remain unchanged. NO proposed code was imported, compiled, syntax-tested, executed, built or run on a GPU. Live tools and calibration002 controller/83binding manifest are unchanged. No calibration outcome files were read and no pass/adoption is claimed.

## F1: preserve actual raw counter identity before post-gate failure

For explicit new-method profiles only, immediately after source-defined run() own-session quiescence the collector hashes the existing complete counters.csv and records its actual byte count, process outcome, capture time and presence in counter_quiescence_snapshot.json. Full raw counter bytes remain in the normal retained counters.csv; there is no wasteful second copy and no synthesized/truncated counter file. This is identity of actual available bytes, not a completeness, numerical or metric certificate.

Source bindings and the method SDK/review are then rechecked in a separate mandatory post-profile check. A rejection writes profiler_method_post_profile_check.json and preserves profile_method_post_profile_validation_hold even through outer exception handling. It never derives/adopts positions. The existing raw process returncode/stop_reason remain unchanged and the full raw CSV/logs remain available. This differs from the historical weakness where a post-run rejection could precede counters_sha256 and later produce an audit KeyError. No old failure/source/receipt is edited or retrospectively certified.

The frozen gate requires both receipt hashes, exact saved process outcome, complete retained CSV SHA/length agreement and passed postcheck for new-method admission. Summarizer freeze includes both receipts and supplies full counters to the gate. Saved rejection becomes the existing per-attempt method validation hold; it cannot interrupt unrelated attempts by the specific missing counters_sha256 path addressed here. Existing audit source is byte-identical to v3: it independently hashes raw counters, then the frozen gate rejects an unapproved attempt and requires it not selected. Arbitrary I/O/corruption failures remain genuine hard integrity failures; no blanket exception bypass is added. Missing/incomplete snapshot/receipt fails closed, not zero-data success.

## F2: immutable calibration orchestration/guard input-source archive

The packet copies only the exact historical input-source bytes whose hashes are already in the unchanged002manifest: calibration_collector_source.py =32a9cfeba983dc3ae8e5d9b8bb85ca4e7736821be0a9c206bb57b80c7471da4c (55,309B); calibration_write_guard_source.py =a2007386239acd7ba26ebcf64aa211756d759570ec8ee0d1a15ccccc25fbc2d3 (4,127B). calibration_input_archive.json preserves original paths/roles. These files are input-source archives, not newly executed code, recovered runtime events or new imports.

Root-only pack_method.py now includes adjacent archived bytes in the method bundle after checking them against83manifest bindings. validate_saved independently repeats both SHA checks and collector identity join. It NEVER executes/imports these two archived sources. Only the unchanged existing calibrated science parser/pure controller validators retain their previously reviewed CPU validation role. The002manifest/controller, exact nine canaries, physical packedhalf conventions,27metric inventory, version payload rules and arithmetic derivation are not changed.

## Source bases / independent axes / later adoption

minimal.patch is complete current-live-base to proposed-v4; v3_to_v4.patch is the small incremental delta. authoring_bindings.json and coexistence_source_review.json identify all bytes. Live bases still collector32a9 / summarizer505d / audit28fa; v4audit equals v3audit. Exact ALLREDUCE admission bodies and selection score remain verbatim; profiler_method cannot authorize the separate2022 two-device collective method.

Root must wait for actual calibration002 finalpass, inspect allnine row/fullhost/identity/metric results, write its concrete scientific review note, run the new private v4 packer to a fresh output and review the resulting bundle/receipt. The packer remains UNEXECUTED. Suggested command (only after actual pass, use exact real completed002 root, no placeholders as evidence):

```
python3 /tmp/roofline-completion-20261021/profiler2025_method_patch/v4/pack_method.py \
 --calibration-root ACTUAL_COMPLETED_002_OUTPUT \
 --source-manifest /tmp/roofline-completion-20261021/profiler2025_calibration002/source_binding_manifest.json \
 --root-review-note ACTUAL_ROOT_SCIENTIFIC_REVIEW_NOTE \
 --output-root FRESH_METHOD_BUNDLE_DIRECTORY
```

Only then may root apply reviewed source bytes and explicitly supersede/readopt the current pinned supervisor frontier with new hashes. Pending002 still pins live collector32a9: do NOT change it while calibration runs. Three-field CSV stratum contract and separate presentation-v2 proposal remain unchanged. No source timing, numerical contract, primary data, ROI phase or causal claims are changed. Profiler launcher runtime remains a separate method stratum; matching launch env does not prove targetDSO equality. GPU5 remains unauthorized.
