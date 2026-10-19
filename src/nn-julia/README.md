# NN: planar nearest-record distance

This named five-size fixture verifies the original Float32 CUDA distance kernel against CUDA.jl, KernelAbstractions and JACC, with independent full-output geometric and record-permutation checks. It is a conditional repaired-reference result: the original CPU record swap aliases its temporary, loses record IDs and misassociates distances. `native_quality_witness.json` preserves actual original full-CLI output and the six-record diagnostic. `corrected_selection.h` uses a Record value temporary; the original CUDA tree and device body are unchanged. These times are excluded from untouched-native aggregates.

The metric is planar Euclidean coordinate distance, **not great-circle distance**. The original external weather database is unavailable. This fixture does not certify that dataset, parser or application quality.

## CPU fixture regeneration

Use Python3.12.3 (stdlib only). From the source repository root:

```sh
python3 src/nn-julia/generate_assets.py /tmp/nn-regenerated
```

Compare every file against `generation_reproduction.json` and `study_plan.json`. The generator creates input_f32.bin, oracle_f64.bin, oracle_exact_u8.bin, study_cases.h, study_cases.jl and study_plan.json. An actual isolated regeneration reproduced all six bytes/hash exactly. The oracle uses independent Float64 `math.hypot` and global sorting; complete ID permutation expectations are constructed from desired IDs rather than source Float32 min scanning. A different Python/libm implementation must verify the published hashes before use. Copy generated assets into this directory only when reconstructing missing assets; preserve archived measured assets.

The three essential CPU-generated binary assets total9,608,354 bytes. Actual GPU captures remain local under `.porting-state`; hashes are provenance references, not a recovery mechanism if those captures are absent.

## Fresh four-way measurement

Pinned Julia and GPU package environments are `src/_julia_ka_env` and `src/_julia_jacc_env`. See `environment_snapshot.json`: Julia1.12.6, CUDA.jl6.2.1, KA0.9.45/JACC1.4.0, CUDA runtime12.0, V100sm70. Native uses nvcc12.0.140. All output allocations, transfers, checks and CPU selection are outside the timed region.

```sh
python3 src/nn-julia/prepare_cold_config.py --output /tmp/nn-cold-config.json
python3 tools/verify_portable_julia.py --config /tmp/nn-cold-config.json nn
```

The preparation script removes all saved-model reuse and relocates absolute source/project paths. The verifier serializes physical GPU6 internally; do not wrap it in an outer flock. It builds native code in isolation and runs each model once, with an explicit entire-R10000 warmup followed by five checked trials. Discard only the first time and average trials2–5. Each trial reports the **sum of five per-case launch-loop means**, not a single-size or whole-application runtime. N=1,17,65,4099,131072. Timed work includes launch-loop overhead and whole-device completion.

All20 measured captures contain947,298 finite values each, including541,376 exact actual-device input-prefix values. Full final GPU distances, full CPU record/distance permutations, scalar input echoes and guards are checked. Repeated launches overwrite distances; no claim of capturing every overwritten intermediate is made. The distance tolerance was declared before execution: atol1e-5 +8eps(Float32)*abs(reference); input bits, IDs, guards and zero/axis controls are exact.

`saved_output_audit.json` records an additional independent squared-norm calculation from actual device-input readbacks, complete ID bijection/association and all20 saved-artifact rehashes. Re-run `audit_saved_outputs.py` only when its archived result and captures are available locally. This audit adds no GPU timing series. NumPy2.5.4 is used by this supplemental audit, not the stdlib generator.

## Interpretation

CUDA.jl, KA and JACC use their own framework device kernels,64 threads/block, no authored shared-memory tile or atomics. One additional untimed parameter observer per implementation checks device scalar inputs. The largest logical input/output footprint is1.5MiB, below V100 L2 capacity6MiB; actual cache behavior, register use and roofline position were not profiled. Translation windows include inherited session/cache/context. CUDA.jl creates the shared harness, KA/JACC reuse it; per-target counters are not independent exclusive billing costs.

See `paper_observations.json`, `scientific_witness.json`, `domain_snippet.json`, `ready_support_files.json` and the per-target calibration/attempt manifests for scope and effort details. JACC's first host-program candidate failed a missing-whitespace generator parse before device JIT; the second candidate changes only whitespace, not device bodies.
