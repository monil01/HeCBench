# BiCGStab Julia study — conditional named solver fixture

## Measured contract

This is a **zero authored GPU kernel** source: cuBLAS vector/reduction calls, cuSPARSE SpMV, ILU0 preparation and triangular solves. All three Julia versions translate the actual host-controlled solver loop in `study_common.jl`. The narrow C ABI exposes individual operations; the Julia ports do **not** call the opaque native solver. KA/JACC initialize the CUDA backend outside the ROI; they do not supply portable solver kernels here. Classification: `hybrid_host_control`, GPU component `vendor_library_interop`.

Named Float64 symmetric MatrixMarket fixture: 32×32 grid, N=1024, E=4992, diagonal4.25/offdiagonal−1, manufactured x*=.75, source max100 and relative tolerance1e−10. The original external TEM27623 matrix is absent and unverified. Source-specific reset P=R at every iteration is preserved, without substituting textbook BiCGStab.

**Conditional native reference:** `native_solver.inc` is extracted from the unchanged native function with (1) the otherwise uninitialized SpSV analysis scalar initialized to1 and (2) one host iteration-counter assignment. The actual solve coefficient was already1. The original CLI succeeds on this fixture; no observed wrong solution is attributed to the undefined analysis scalar. Full original native source stays unchanged. This conditional defined-input reference is excluded from untouched-native aggregates. All workspaces are zero-defined and guarded outside the timed region for complete state capture.

The timer includes SpSV descriptor creation, buffer allocation, analysis, complete dependent host loop, final true-residual norm, descriptor destruction/free. ILU0 preparation, capture, and all checks are outside the ROI. Whole-device completion before/after. Each model warms the exact full solver, then runs five checked trials; mean trials2–5. Native time is recorded inside the C++ adapter; Julia time surrounds the translated loop. Differences include real host/FFI boundaries. No kernel-only or causal abstraction ranking is justified.

## Evidence

Final bundle: `.porting-state/portable_julia/runs/20261019T175356789956Z_bicgstab/result.json`.

All20 actual captures are bitidentical: 55,236 Float64-encoded values each, including 27,617 exact input-prefix values (220,936 bytes), CSR/factors, all11 initial/final vector buffers, guards, final norm and iteration count77. The input prefix comes from actual device readbacks, not host shadows. Final surviving states are captured; overwritten per-iteration states are not claimed.

Independent C++ coordinate-stencil/row-elimination checks and separately expressed Python `math.fsum`/coordinate/map-ILU0 saved-output audit pass. Unique solution follows from strict diagonal dominance margin.25. Actual solution maxerror2.35e−10; true residual≈1.052e−9. Predeclared independent limits: solution2e−8, residual2e−9, stored residual/returned norm agreement1e−12, ILU0 discrepancy1e−12. Full dual numerical contract atol1e−7/rtol1e−7; observed differences zero. All state values finite and all guards exact.

`saved_output_audit.json` hashes11 source bindings, all20 artifacts, four exactly regenerated source/fixture files and actual loaded DSOs. All four use identical cuBLAS12.0.2.224 and cuSPARSE12.0.1.140 binaries. Vendor launch geometry, registers, occupancy, internal compilation and causal bottlenecks remain unknown.

Mean seconds (trials2–5): native.04955898975, CUDA.jl.052695834, KA.04840410125, JACC.0478920455. Single named matrix, sequential series, full host/library scope; do not generalize to original TEM production throughput.

Each target: one authored host-solver candidate, one host Julia JIT process check, zero authored device definitions and zero actual authored-device JIT checks. The absent launcher invocation was interrupted while queued before Julia started. Native setup failures and the saved-log-lineage/annotation lookup mistakes are recorded separately as tooling events. Current observed token windows include context, cache and coordination; native preparation and CUDA.jl shared-harness creation are separated. See per-target manifests/calibration and `paper_observations.json`.

## Cold reproduction

From the source repository root, regenerate the small text fixture and exact extracted bridge:

```sh
python3 src/bicgstab-julia/native_diagnostics/generate_readiness_matrix.py src/bicgstab-julia/study_grid.mtx
python3 src/bicgstab-julia/generate_bridge.py
```

No irreplaceable CPU binaries are needed. `generate_bridge.py` requires the original `src/bicgstab-cuda/{main.cu,utils.h}`. Keep the original source notice in `LICENSE.source`. Installed requirements: Python3.12.3 stdlib, native nvcc12.0.140/cuBLAS/cuSPARSE, pinned Julia1.12.6/CUDA6.2.1/KA0.9.45/JACC1.4.0 projects, local CUDA12.0 and an NVIDIA driver. No PyTorch or external TEM dataset is used. Actual library versions/hashes are in the audit; a different installation defines a new environment stratum.

For a fresh relocated measurement, use `prepare_cold_config.py --source-root /path/to/source --output /tmp/bicgstab.json`, then `python3 tools/verify_portable_julia.py bicgstab --config /tmp/bicgstab.json`. This removes old-series reuse. The verifier acquires the shared GPU lock internally; do not wrap it in another flock. Original pinned commands use GPU6.

Read-only saved-evidence verification, requiring existing local artifact paths:

```sh
python3 src/bicgstab-julia/audit_saved_outputs.py .porting-state/portable_julia/runs/20261019T175356789956Z_bicgstab/result.json
```

Raw GPU captures and old executables remain local, not committed. Exact cold regeneration proves CPU fixture/source reproduction, not recreation of historical timings. All canonical traits/status/calibration files live under `.porting-state/portable_julia`; the root coordinator exports them and commits each verified version separately.
