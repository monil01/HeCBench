# HEARTWALL: complete GPU tracking kernel

## Verified named contract

`heartwall_full51points_11realAVIframes_all37GPUstages_templateupdate_v1` runs the unchanged native `hw` kernel and full CUDA.jl, KernelAbstractions and JACC translations. Original 51 points (20 endocardial, 31 epicardial), 744×656 real AVI frames 0–10, 51×51 templates, 81×81 search areas, source 51 blocks ×256 threads. The periodic template update at frame10 is included. “37 stages” describes source conceptual work inside **one kernel definition**, not 37 kernels or translation attempts.

Final receipt: `.porting-state/portable_julia/runs/20261019T214127826428Z_heartwall/result.json`. All four models have five checked in-process trials, exact input-prefix bytes, full 26,373,184-Float32 captures/trial and the same eleven kernel/completion timing regions. Frame H2D, readback, snapshots and CPU checking are outside the timers. Mean trials2–5: native39.2288ms; CUDA.jl50.070593ms; KA49.50913375ms; JACC49.2879225ms. JIT/full-pipeline warmup precedes trial1.

All30 global arrays and complete coordinate histories are captured. Independently expressed Float64 FFT convolution/window integrals validate full NCC/box statistics and all561 row plus561 column coordinates exactly; final30 fields use explicit per-element operator budgets. Dual GPU budgets: atol.002, rtol2e-6; independent NCC atol1e-3, final template3e-5, exact box sums/coordinates/guards. The NCC ceiling was informed by native readiness diagnostics and frozen before primary trials. Only two template-blend arrays differ across models, max1.52587890625e-5; other28 arrays are bit identical. The raw structured max_abs_error≈159.96 is the maximum over **mixed-unit independent scalar fields including large covariance**, not NCC error. max_rel_error is uncomputed/null.

Each implementation additionally executes a separate actual untimed 11-frame replay, capturing all30 post-kernel global fields/frame. These are not overwritten intra-kernel/timed intermediate observations. Independent checks cover NCC/covariance/box sums/templates/coordinates/guards every applicable frame and all30 operators at the final frame; the remaining saved global arrays are fully compared across models, with no claim that every overwritten one of37 stages was independently observed. `saved_output_audit.json` checks all30 primary pair/trials and all66 replay pair/frames. Raw GPU files remain local, referenced by hash/path.

## Scope and implementation

Original CUDA sources are immutable. The actual unmodified `./main 11` CLI was executed separately using the source-declared real DVC dataset. The primary is an integration-adapted resident study harness with defined zero scratch/future tracking slots and identical 16-element end guards; native arithmetic is unchanged. It is excluded from untouched-native aggregates. Source host AVI decode/CLI integration is not translated; all104 frames and clinical tracking quality remain unverified.

Source denominator sqrt((n−1)*population_variance), absolute NCC argmax/first-columnmajor tie, and even-width support[-5,+4] are preserved method policies, not automatically scientific defects. All targets use their own framework kernel/barrier APIs. KA uses CUDA-backend CuArray allocation; JACC uses its own array/parallel_for and explicit shmem_size0. Standard Float32 fma/sqrt are backend-lowered; no foreign CUDA device intrinsic, library substitute or CPU computation replaces GPU tracking stages. All30 global intermediate layouts and sourcegeometry are retained through typed zero-based array views. Native74 registers/no spills is compiler evidence; target registers/occupancy/bandwidth are unknown, with no roofline cause inferred. Wrapper allocation, descriptor ABI, launch dispatch, guards and input readbacks remain timing covariates.

## Cold reproduction without committed large binaries

Use existing g++/NVCC12.0/Julia1.12.6 and pinned study projects. CPU oracle requires **NumPy2.5.4**, recorded separately from any comparator version. The original source AVI helpers are used only for CPU file decoding. Source extraction uses existing isolated **pycparser2.22**; it is not a runtime dependency of the ports.

```sh
python3 src/heartwall-julia/fetch_dataset.py --output-dir /tmp/heartwall-dataset
PYTHONPATH=/tmp/pasc-julia-plot-tools python3 src/heartwall-julia/cold_regenerate.py --data-dir /tmp/heartwall-dataset --materialize
PYTHONPATH=/tmp/pasc-julia-plot-tools python3 tools/verify_portable_julia.py --config src/heartwall-julia/study_config.json
```

`cold_regenerate.py` first generates into an isolated temporary tree and checks exact hashes against existing assets or `asset_manifest.json`; `--materialize` copies only missing CPU `.bin` files after successful checks. The original cold audit reproduced34 assets byte-for-byte. The source configs contain absolute workspace paths and archived reuse receipts: relocate those paths and remove `reuse_result`/`reuse_models` for a fresh all-four study. Each child needs runtime-asset symlinks `../<asset>` for names in its model spec; these are local regenerable fixtures, not required Git binaries. The runner acquires the GPU6 study lock internally; do not wrap it in another flock.

`translate_kernel.py cuda_jl|ka|jacc` regenerates complete device bodies from the source C AST. It removes only disabled TEST_CHECKSUM instrumentation, preserves C Int32/Float32 types, zero-based pointer offsets, integer truncating division, original branches/loops/barriers, and explicit contracted product-add arithmetic. Historical CUDA.jl/KA generator bytes and hashes are retained in `translation_generators/v1_cuda_jl_ka.py`; final generator differs only in prospective JACC index emission. Target entry files themselves are the verified immutable code.

## Effort and tokens

One authored GPU candidate and one actual device-JIT process per target. Two CUDA.jl runner runtime-asset staging configuration failures occurred before Julia/GPU execution and are separate setup costs. Shared translator, device views and CPU/capture harness creation is charged to the CUDA.jl window; KA/JACC reuse is ordered. Prospective actual context-inclusive input/cache/output/reasoning counters, model metadata and calibration adjudications are in canonical STATE paths. Cached/reasoning counts are subsets, not extra tokens; these are not exclusive billed cost or recoverable historical consumption. Native preparation, verification and annotation are separate. Shared verification END was captured in the same tool round as the last source-binding write, so that window conservatively excludes some final-generation activity; the original event and scope limitation are retained, never backfilled. Target translation ENDs were separate rounds.

Original diagnostic wrapper captures were inserted before its end timestamp; its reported diagnostic times include captures and are invalid for performance. Only the five checked primary study timings above are used. All source witnesses, failed setup logs, raw captures and observed boundaries are preserved locally by hash/provenance.
