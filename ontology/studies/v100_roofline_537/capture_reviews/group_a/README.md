# Capture overlays A

Source helper `tools/v100_roofline_prepare_capture_overlays_a.py`; final hash and17staged plan hashes in delivery.json. Cases: clockJulia3,linearprobing4,pointerchaseJulia3,RSCJulia3,RSMT4. Only this new helper written in source; frozen other helpers/ledgers/accepted sources/results/configs untouched.

```sh
python3 tools/v100_roofline_prepare_capture_overlays_a.py \
 --base-plan-root /tmp/roofline-method-review/reviewed_plan_v4_plus_v5_cap512 \
 --reviews /tmp/roofline-method-review/dynamic_output_scan/confirmed_output_root_proposals.json \
 --reviews-sha256 24bfd2759ffd456d5bff8a801560006a84d4feec674e625106ce59e83f193d66 \
 --output-root /tmp/NEW_CAPTURE_A_PLANS --stage
```

Fresh/tmp root only; default512MiB declared assets/model and2GiB remainingreserve. Optional repeatable --benchmark. Without--stage only inventory. The frozen host overlay helper is imported for standard-library hash/path/check/dump utilities, with exact8f940...9ceSHA verified and included in every execution binding; import has no compiler/target calls and does not write bytecode. Changing that dependency requires review/regeneration, never silent fallback.

## Exact allowed changes

- One source-root-derived capture assignment per common/native adapter → fresh model STUDY_OUTPUT_DIR/private_CASE (native equivalent absolute scratch path).
- Copied Julia entry includes the copied common; every other directory-relative include/fixture/library reference retains canonical original source location. RSC FIT_LIB is readonly and is not redirected. RSMT CHECKLIB stays cwd studycheck.so, built fresh by unchanged flags.
- Copied native translation unit quoted includes refer to canonical originals. Original kernel body definitions are included unchanged. No geometry, data, RNG, iteration, acceptance, queue ordering, clock instrumentation or arithmetic edits.
- Same compiler flags/input source semantics; only compiled source copy/output path changes. Every build wrapped in existing hash-bound child Landlock allowing writes only in scratch. No compiler invocation was performed here.

## Current evidence

inventory_v2:17/17ready, unstaged. staged_v2:17/17ready/staged, alloriginal/copy hashes and source-before/after equality rechecked. Each model has plan_diff.json and complete exact original/config/result/review/source bindings; compiler_manifest.json where applicable. Earlier conservative metadata-only inventory naming failure retained, not counted as benchmark/compiler attempt.

Readiness concerns filesystem output isolation only. Actual full checks, writeguard, source rehash under GPUlock, profiler errors, correct timedROI/phase mapping and replay qualification remain mandatory. Pointerchase's execution-only scope and absence of numeric outputs remain unchanged; none of these path changes authorizes an arithmetic roofline category. Existing historical capture bytes must remain untouched.
