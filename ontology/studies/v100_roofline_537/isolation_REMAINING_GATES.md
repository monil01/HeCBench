# Remaining roofline preparation gates

Read-only review; no source/shared metadata changes, builds, target execution, GPU work, or staging. Existing accepted benchmark evidence is unchanged.

## Counts

- Frozen v4 plus 512 MiB per-model asset cap: 2,013 ready / 115 blocked, plus 20 terminal hardware slots.
- Additive v5: 16 models independently reviewed. Planner-only merged regeneration: **2,029 ready / 99 blocked**, plus the same 20 terminal slots.
- Ready means isolated input/build plan reviewed; exact timed-ROI mapping, kernel census, replay safety, counters, and measured ontology remain separate.

| Category | Slots |
|---|---:|
| actionable_asset_budget_and_output_overlay | 1 |
| actionable_build_and_runtime_side_effect_review | 11 |
| actionable_readonly_tool_role_review | 6 |
| actionable_runtime_path_review | 19 |
| actionable_source_output_overlay_required | 29 |
| approved_additive_v5_cap512 | 16 |
| multi_device_contract_outside_gpu6_only_campaign | 20 |
| proposal_requires_new_executor_support | 13 |

## Executor proposals (not approved)

See `../overlay_proposals/action_proposals.json` and `profile_build_overlay.py`. They are proposals, not executable approvals or tested builds. XLQC requires a writable fresh clone of its fixture tree: parse_basis writes derived basis files. Interval native requires a precreated fresh child directory; SSS must receive a nonexistent output root. Pad must avoid the shared CPU-component cache, including cache-hit metadata writes. Leukocyte must compile Meschach in a fresh copied tree, never make in source.

## Asset budget / read-only link assessment

Raising cap64→512 MiB removes25/26 budget gates but initially unblocks only13 models; other source/output gates still apply. SU3 remains673,919,456 B. Its input reader is readonly, making a future read-only symlink input plan plausible, but current planner does not implement that review and a separate historical failure dump still blocks it. No symlink workaround was staged. Other below-cap assets may be copied into the fresh scratch only after global free-space/reserve checks. No historic data were moved or deleted.

## Classification and scope

Twenty multi-device model slots are not eligible for a GPU6-only whole-contract run; they are concurrency/resource constraints, not framework limitations. Twenty additional slots correspond to five already evidenced native hardware barriers and remain unavailable, without alternate arithmetic. Runtime-only/empty, integer and SFU kernels can be executable yet have floating roofline positioning not applicable or incomplete. Do not confuse such metric applicability with preparation readiness. For host/device shared-flag or MPI work, profile replay can change synchronization and must be qualified separately. Every remaining gate is either an actionable isolation/review item or a multi-device contract; none is evidence of framework unsupported functionality.

## Per-case actions

- **allreduce** (cuda, cuda_jl, jacc, ka): Reserve both required devices and retain distributed communication ROI; never silently reduce device count.
- **bn** (cuda): Fixed historical failure dump must be redirected even if only reached on mismatch.
- **ccl** (cuda, cuda_jl, jacc, ka): Reserve both required devices and retain distributed communication ROI; never silently reduce device count.
- **clock** (cuda): Fixed raw timings destination requires scratch route; do not discard timing evidence.
- **columnarSolver** (cuda, cuda_jl, jacc, ka): Native wrapper creates /tmp/columnarSolver_native/checked_PID outside owned scratch; target argv points at old output roots. Review fresh root override, not input role.
- **easyWave** (cuda): Literal remains unreviewed. Trace exact active writer/read-only call before approval.
- **gc** (cuda, cuda_jl, jacc, ka): Fixed shared raw sidecar root in study_common; requires profile-specific source overlay retaining original hashes/diff.
- **gibbs** (jacc): Explicit final trace directory argument; inspect actual writer and redirect to scratch after review, preserving revised per-field scientific contract.
- **grep** (cuda, cuda_jl, jacc, ka): Explicit private trace directory arguments; prove all writer paths under argv root before review approval.
- **halo-finder** (cuda, cuda_jl, jacc, ka): Review compiler TemporaryDirectory/output behavior and configured cudaUtil.h /tmp/nvidia-mps_rank setup; MPI/MPS process context and read-only parent inputs not yet proven isolated.
- **heartwall** (cuda, cuda_jl, jacc, ka): Asset budget now clears (534656324B <512MiB). Explicit trace root still needs full output writer review.
- **interval** (cuda): Native raw folder must be created fresh before execution, exact argv1 replaced. Existing native source does not mkdir. Never apply generic precreation to SSS (requires nonexistent output root).
- **leukocyte** (cuda, cuda_jl, jacc, ka): Native original builder runs make in source Meschach tree: unsafe. Proposed fresh tree + objects/archive compile. All models still need exact private output root review; cap512 required.
- **logic-resim** (cuda, cuda_jl, jacc, ka): Apply exact hash-bound supplemental review; planner-only validated, no build/profile executed.
- **memcpy** (cuda): Fixed /tmp/memcpy_PID_trial.bin opens without exclusive creation; PID alone is not isolated scratch.
- **minibude** (cuda_jl, jacc, ka): Nested Julia executable argument needs reviewed executable role and wrapper side-effect inspection; no reason to treat compiler/runtime path as output.
- **p2p** (cuda, cuda_jl, jacc, ka): Reserve both required devices and retain distributed communication ROI; never silently reduce device count.
- **p4** (cuda, cuda_jl, jacc, ka): Fixed historical raw sidecar paths require explicit source overlay, not blanket absolute-literal waiver.
- **pad** (cuda, cuda_jl, jacc, ka): Old build_study.py mutates shared CPU-component cache metadata even on cache hit. Proposal emits same compilation into fresh scratch and cwd cpu_component_path.txt.
- **particlefilter** (cuda): Fixed /tmp/particlefilter_PID_trial.bin writer needs scratch source overlay.
- **pingpong** (cuda, cuda_jl, jacc, ka): Reserve both required devices and retain distributed communication ROI; never silently reduce device count.
- **pointwise** (cuda): Fixed /tmp/pointwise_PID_trial.bin writer needs scratch source overlay.
- **rayleighBenardConvection** (cuda, cuda_jl, jacc, ka): Apply exact hash-bound supplemental review; planner-only validated, no build/profile executed.
- **recursiveGaussian** (cuda): Fixed /tmp/recursiveGaussian_PID_trial.bin writer needs scratch source overlay.
- **remap** (cuda, cuda_jl, jacc, ka): Apply exact hash-bound supplemental review; planner-only validated, no build/profile executed.
- **ring** (cuda, cuda_jl, jacc, ka): Reserve both required devices and retain distributed communication ROI; never silently reduce device count.
- **rowwiseMoments** (cuda): Fixed /tmp/rowwise_PID_trial.bin writer needs scratch source overlay.
- **sc** (cuda, cuda_jl, jacc, ka): Apply exact hash-bound supplemental review; planner-only validated, no build/profile executed.
- **slu** (cuda, cuda_jl, jacc, ka): Builder acquires global CPU cache build.lock and may build archives there. Fresh cache compiler overlay plus readonly fixture and private argv review required; source GPU algorithm unchanged.
- **sparkler** (cuda_jl, jacc, ka): Nested Julia executable role unresolved. Must inspect MPI wrapper and cwd/output effects before approval; wrapper children need target-processes all.
- **spsm** (cuda, cuda_jl, jacc, ka): Fixed warmup failure dump in historical review directory; preserve failure path evidence with fresh destination.
- **sssp** (cuda, cuda_jl, jacc, ka): Explicit private output directory needs writer-only proof and exact scratch argv replacement.
- **stencil1d** (cuda): mkstemp atomically creates fresh /tmp/hecbench-stencil1d-XXXXXX (not overwrite risk), but outside designated scratch. Either explicit fresh-file exception+cleanup manifest or source route required.
- **stsg** (cuda): Fixed historical failure-only warmup dump still violates isolation when profiling fails.
- **su3** (cuda): 673919456B assets exceed512MiB. readv uses readonly ifstream, so future readonly symlink inventory is plausible, but fixed historical failure-dump writer independently blocks; do not waive both gates together.
- **svd3x3** (cuda_jl, jacc, ka): Julia specs build -o into historical source path; redirect compiler output AND dependent shared-library argv, plus nested Julia executable role. No blanket approval.
- **tsne** (cuda, cuda_jl, jacc, ka): Fixed private PID output roots and run_metric.py capture copying require fresh source-overlay routes; readonly oracle path role alone cannot clear writer gate.
- **voxelization** (cuda, cuda_jl, jacc, ka): study_capture.py uses fixed state root with PID/timestamp copyfile; uniqueness is not guaranteed exclusive creation. Route all copies under new scratch.
- **wmma** (cuda): Fixed historical geometry diagnostic output needs reachability review or scratch route. Preserve alternate compiler/header/static-runtime stratum.
- **xlqc** (cuda, cuda_jl, jacc, ka): Source chdir(fixtures), parse_basis writes basis_<element>.dat/basis_all.dat. Must clone full small fixture tree fresh+writable and replace exact fixture/private argv; never mark whole old tree readonly.

Machine-readable table binds each source plan/result/config hash and retains the original literal findings. Exact source evidence and execution binding hashes for approved cases are in the immutable reviews; proposal evidence is in action_proposals.json. No claim that unreviewed case actions have already proved isolation.
