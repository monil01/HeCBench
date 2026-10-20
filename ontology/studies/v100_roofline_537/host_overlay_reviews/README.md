# Host/output isolation overlays

`tools/v100_roofline_prepare_host_overlays.py` writes only fresh `/tmp` plans and optional copied inputs/sources. No compiler, target, CUDA initialization or GPU invocation. Existing helpers, kernels, accepted configs/results, historical captures and Git records unchanged. `delivery.json` pins final helper, guard, review tables and eleven staged plan hashes.

## Usage

```sh
python3 tools/v100_roofline_prepare_host_overlays.py \
 --base-plan-root /tmp/roofline-method-review/reviewed_plan_v4_plus_v5_cap512 \
 --proposals /tmp/roofline-method-review/remaining_runtime_paths/new_plan_proposals.json \
 --proposals-sha256 3ae123c52266747a984a98875f36f51f2a3d22b7fef540e6b1406901a7988785 \
 --roles /tmp/roofline-method-review/remaining_runtime_paths/nested_executable_role_reviews.json \
 --roles-sha256 92b02864d7b84294decb7b6cbbd5ec87af890bba7e4d3a008cf204718d0d045f \
 --output-root /tmp/NEW_HOST_OVERLAY_PLANS --stage
```

Optional repeatable `--benchmark`. Default512MiB assets/model and2GiB remaining-space reserve. Existing output root and source/config/result/review/hash drift are fatal. No fallback weakens isolation. Source before/after equality and exact copied source bytes are rehashed on staging; collector must repeat runtime integrity checks. Default without `--stage` emits inventory only and is not executable by collector.

## Cases

- Hungarian Julia3: copied common redirects auxiliary raw capture root below STUDY_OUTPUT_DIR. Canonical fixture/include roots are retained explicitly. Copied entry changes common include and source-root lookup only. Managed flags, all16solver stages, arithmetic and kernel bodies unchanged.
- MiniBUDE Julia3: only inline STUDY_RAW_DIR assignment changes to new scratch/profile_raw (precreated). Canonical entry/includes and every kernel/env numeric knob unchanged. Nested absolute Julia executable remains the original path; its symlink, resolved launcher and bytes have an explicit readonly executable binding. This does not change Juliaup version/channel or claim a new interpreter version measurement.
- Sparkler Julia3: copied common retains canonical ROOT and cwd study_backend.so while CAS/ledger capture root moves below STUDY_OUTPUT_DIR. Copied entry's common include changes only. Original MPI4 ranks onGPU6, communicator, collectives, streams, kernel/vendor arithmetic and backend build flags stay intact. MPI temporary transport files may require additional explicit scratch configuration if Landlock rejects their default locations; do not widen historical write access or silently change communication algorithm. Path readiness does not authorize MPI replay safety.
- Native columnarSolver: copied host wrapper keeps canonical P for source hash audit, makes new output-private root exclusively and invokes a new scratch binary. Readonly CPUgold path stays original. Adds accepted README CUDA12.0 compile flags, canonical source, exact DATA_PATH macro and fresh output; no source/kernel edits. This preserves the preexisting corrected-native mathematical qualification.
- Native stencil1d: copied source changes only mkstemp template to new scratch and original include to its canonical location. Exclusive temporary-file creation and full output bytes remain intact; no historical temporary artifact is removed. Original compiler flags/geometry/trials unchanged.

## Build enforcement and limits

Every nonempty build command is wrapped by the existing source-bound `v100_profile_write_guard.py --allow-dir SCRATCH -- ...`; the current collector also confines child writes. Compiler command/guard hashes are recorded in compiler_manifest.json. No build command was invoked in this work. Build outputs are fresh; source includes/data remain readonly. Profiler execution requires the collector's guard and usual lock. Replay safety, full scientific acceptance, exact timed ROI mapping, meaningful memory tier and aggregate eligibility remain separate from path readiness. StreamUM stays held unknown; this helper does not handle or authorize it.

`inventory_v1` has11/11 static-ready plans; `staged_v1` has11/11 staged plans, hashes checked, with no compilation or GPU execution. Each plan retains exact source/config/result/proposal bindings and plan_diff.json. No attempt or compiler-feedback count is added to benchmark translation effort: this is separately metered shared measurement infrastructure.
