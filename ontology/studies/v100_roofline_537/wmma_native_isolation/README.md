# Native WMMA isolation proposal

The selected native adapter has one absolute path literal: the excluded geometry diagnostic directory at study_native.cu line20. It writes impl1_96x80_output_f32.bin and native_geometry.json there before scientific warmups. Input/oracle files are relative read-only fixtures already declared in the plan. Trial outputs use STUDY_OUTPUT_DIR.

`prepare_native_wmma.py` is a **not-executed CPU staging helper**. Root can run:

```sh
python3 /tmp/roofline-method-review/wmma_native_isolation/prepare_native_wmma.py \
  --base-plan /tmp/hecbench-roofline-v100-20261020/batch017-wmma-typed/wmma/cuda/plan.json \
  --output-root /tmp/hecbench-roofline-v100-20261020/batch018-wmma-native-isolated \
  --stage
```

Choose another fresh output root if that name exists. No build or GPU command is run by this helper. It verifies exact selected source/config/result hashes, refuses unrelated blocking gates, preserves CUDA12.9 nvcc/-O3/-std=c++17/-arch=sm_70, and redirects exactly one host path literal in a private adapter copy. Native kernel source/reference/case-header copies remain byte-identical. The inverse replacement reconstructs the original adapter bytes. A diff and all copied-source hashes are retained.

The diagnostic still runs and writes inside the fresh private attempt. Its metadata output_path changes accordingly. All fixtures, warmups, eleven-case R100 trials, full-output/guard checks and numerical code remain. Keep the existing write guard; do not grant write access to the accepted diagnostics directory. The helper clears only the reviewed literal-isolation gate, and --stage is required before collector execution is possible.

## Tensor integration source review

The loader/classifier canonical mode mismatch is resolved in the current source snapshot. The collector forwards per-row approvals, exact selected hashes and prospective dependency bindings; no queued-profile success is inferred here.

One source mutation path remains: tensor loader source snapshot is written before the collector's existing-profile guard. Reinvoking a frozen attempt with a changed loader can overwrite that snapshot before returning existing_profile_preserved. Move that guard ahead of any slot write, or reject differing existing snapshot bytes. No actual overwrite was observed.

For the native overlay, original accepted source hashes alone do not identify the compiled adapter copy. Retain the source diff, helper/source-copy/compiler/binary receipts and independent path-only proof alongside tensor decisions. The source/API mode is unchanged by the diagnostic output-path replacement; no selected target SASS observation is added. Mode eligibility never promotes the excluded geometry diagnostic or warmups into the primary ROI.

`review.json` binds the exact plan/result/config/source and source-only observations. No tests, build, GPU query/execution, accepted-source change or shared tool/metadata edit was performed.
