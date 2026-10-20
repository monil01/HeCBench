# Rayleigh–Bénard convection: full source-method study

Original500×1000 grid, Ra1e8, full1000RK3steps. All original seven CUDA mathematical bodies and the complete cuBLAS/finite-difference/controller pipeline are preserved. Six pointwise entries execute; unused AddX is authored in targets but not numerically certified. CUDA.jl/KA/JACC source children replace every selected pointwise dispatch through an explicit CABI callback. Original CPU orchestration and vendor cuBLAS remain: these are hybrid pipeline translations, not independent GPU-library implementations.

## Scientific contract

Complete actual initialized device inputs and all final mutable working fields, 1000 adaptive steps and all three source heat-diagnostic samples are retained. See verification_plan.md and source_scope_review.json. Exact dual-GPU comparisons are separate from the full independent Float64 source-method RK3 engineering gate (0.3% per-field scaledLinf; actual adaptive history independently predicted). This is not a continuum/discretization convergence certificate.

The source removes the conducting temperature profile before its function called NusseltCompute. Its result is an excess/background-subtracted heat diagnostic, not total dimensionlessNu: the independent equilibrium witness returns0 while total conductingNu is1. Preserve this known diagnostic qualification/exclusion. Also retain source verticalDX for horizontal derivatives although horizontal initialization spacing differs slightly. No silently repaired geometry or full physical heat-transfer validation is claimed.

Original recovered-source provenance is expressly uncertain in its header. Do not invent an associated paper citation. No original array guards exist; complete snapshots cannot establish absence of every out-of-bounds access.

## Clean-checkout regeneration and execution

Use existing Julia1.12.6/CUDA6.2.1/KA0.9.45/JACC1.4.0, nvcc12.0 and the pinned environments. No OS installation or sudo. Create a **new** fixture directory; the preparer refuses overwrite and requires2GiB free reserve. It serializes native compilation/execution under the common GPU6lock and then runs the CPU-only independent oracle. Its native run is a preparation, not a five-trial performance result.

From this benchmark directory:

```sh
python3 prepare_fixtures.py --output /tmp/hecbench-rbc-new-fixtures
python3 conduction_heat_flux_witness.py
julia --startup-file=no spatial_control_witness.jl
python3 make_config.py --fixtures /tmp/hecbench-rbc-new-fixtures --models cuda,cuda_jl,ka,jacc --output /tmp/hecbench-rbc-new-config.json
python3 run_study.py --config /tmp/hecbench-rbc-new-config.json
```

The central verifier owns the GPU6lock; do not add an outer flock around run_study.py. The source-owned launcher uses process-only STATE=/tmp/hecbench-stencil-ports-state and creates canonical result symlinks only after completion. Each model executes full warmup plus five checked trials; mean trials2–5. The ROI includes the complete original loop/cuBLAS operations/host decisions plus common scalar history capture replacing Nuprintf and device-wide completion. Initialization/CPUoracle/checks/outputserialization are outside it. Not a kernel-only roofline measurement.

Run the independent CPU full artifact replay without new GPU trials:

```sh
python3 audit_captures.py /path/to/result.json --fixtures /tmp/hecbench-rbc-new-fixtures --output /tmp/hecbench-rbc-audit.json
```

actual_input_manifest.json records expected byte sizes and SHA256 for the measured-environment inputs/oracles; pin runtime/compiler/libm/CPU BLAS when interpreting exact hashes. Runtime cuBLAS paths/versions/hashes are separately process-observed. Large regenerable arrays and raw captures stay outside source Git. /tmp is local ephemeral storage: preserve research evidence before cleanup/reboot; canonical symlinks are not an off-host archive.

Sources: reference_history retains original main.cu/Makefile/CMakeLists byte-identically; configured_source.cuh is the reviewed dispatch/observer integration, reproducible by configure_source.py. No sourcebody algorithm change is hidden. Accepted sources/configs/results remain frozen; observations/manifests describe qualifications separately.

Source buffer roles are explicit in source_scope_review.json: final `u` contains `Dz(T_stage)*v_stage`, not horizontal velocity; `omega` is the transformed/scaled interior copied into `psi`, not certified physical vorticity. The last G-stage fields precede the final RK3 temperature combination. Read-only transform tables are captured as actual inputs; an exhaustive post-run read-only memory audit is not claimed.

`make run` / `julia main.jl FIXTURE_DIRECTORY` is a CUDA.jl-only checked five-trial convenience entry; it does not perform or claim the four-way comparison. Use `make run-all` or the explicit all-model commands above for that study.
