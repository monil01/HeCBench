# KMC: Gaussian SVM kernel-matrix construction

## Actual work and verified scope

KMC means **kernel-matrix construction**, not kinetic Monte Carlo. The source uses cuBLAS Float32 SGEMV for each training vector, transfers the dot products back, and computes the Gaussian RBF in CPU Float64. It has **zero authored device kernels**. All three Julia versions preserve that vendor/host workflow; KA/JACC use actual own-framework array allocation plus explicitly CUDA-specific cuBLAS integration. No fictitious `@kernel`/`parallel_for` is added or counted. This does not establish portable KA/JACC device code.

Named contract `kmc_realGISSETTE_1_257_513rows_all5001features_fullSGEMV_RBF_pipeline_v2`: real published GISSETTE rows1/257/513, all5000features plus original zero index0 slot,771 sequential row SGEMVs. The original6000-row matrix and SVM/CV training are not translated/certified. Actual unchanged original CLI on first513 rows reported90.6433% CV, a diagnostic rather than independent quality validation.

Final receipt `.porting-state/portable_julia/runs/20261019T215956436446Z_kmc/result.json`. All twenty full8,421,838-Float64 captures pass, including7,737,186 actual-input prefix values. Actual GPU matrix and every row vector are read back after use, before reuse; initial GPU output state is captured. All GPU dots/norms/source rowIDs/guards are bit identical across all models. Only CPU exp differs, max2.220446049250313e-16. Native study513-case dots and full RBF matrix are bit identical to the actual original unchanged `cal_km` diagnostic. Complete independent CPU Float64 Gram/Euclidean-distance Gaussian operators and explicit per-entry roundoff bounds check every output; no checksum substitute.

GPU compute/storage is Float32. Host source norms accumulate Float32 squares into Float64, followed by Float64 exp. Float64 capture encoding preserves actual Float32 values; it is not a Float64 GPU-compute claim. Source diagonal roundoff (~3.39e-7 above1) is preserved and mathematically bounded, not automatically a scientific defect.

## Timing and interpretation

Five checked in-process trials, full-workflow warmup before trial1; mean trials2–5: native36.810615ms, CUDA.jl49.875447ms, KA50.3712375ms, JACC47.9266495ms. Primary ROI sums771 intervals of row H2D→SGEMV→dot D2H completion. Source CPU Gaussian transforms, actual-input readbacks, snapshots and checking are excluded. This matches the source inner offload timing scope; the source separate outer total timer and full SVM driver are not the primary measurement.

All implementations load **the same cuBLAS12.0.2 binary**, hash in `vendor_policy.json`. Native uses32-bit SGEMV entry/host alpha-beta/default math0/legacy default handle stream. CUDA.jl wrappers use64-bit SGEMV entry/device CuRef alpha-beta/math16/task stream. Supplemental fresh environment queries confirm the Julia defaults and are clearly not primary executions. These API/handle/stream and wrapper-allocation differences remain covariates; results do not isolate causal authored-kernel abstraction performance. Guard alignment/readback/cache effects are disclosed; opaque vendor algorithms/registers/roofline/bandwidth remain unknown. The named study is integration-adapted and excluded from untouched-native aggregates, with **no native arithmetic correction**.

## Preserved feedback and revised oracle

The first native build specification used unsupported NVCC `-Wl` syntax; `-Xlinker` fixed that packaging issue before GPU execution. Native v1 warmup then rejected9.09e-13 from comparing sequential Float64 norm addition against NumPy's pairwise sum under an exact-zero budget. The original criterion/generator/config/budget and failed result remain in `native_diagnostics/criterion_v1` and STATE. Before target translation, v2 uses the standard gamma_D(binary64) norm-summation bound. This is an explicitly **revised verification contract**, not a GPU arithmetic repair or framework failure. All five v2 native trials and all targets were measured afresh under that frozen criterion.

CUDA.jl had one host generator-whitespace ParseError before GPU/library execution, followed by a passing second program check; KA/JACC first program checks passed. Authored-device candidates, kernels and device-JIT checks are all zero. Shared vendor-policy diagnostic helper initially used Ref(enum) instead of Ref(UInt32 flags), failing in each project; two corrected supplemental queries succeeded. Those four processes are separate verification/environment cost, not target program/device-JIT attempts. No OS/system package update or sudo command was invoked.

## Reproduction and assets

Source copyright is retained in `LICENSE.source`. Published input is22MB compressed, hash/size checked by `fetch_dataset.py`. Four CPU-generated binary assets regenerate exactly on the recorded NumPy2.5.4/CPU BLAS platform; keep their recipes/hashes, not large binaries or raw GPU outputs, in Git. A different CPU BLAS can change low oracle bits; exact local cold-regeneration evidence and the mathematical bounds are distinct claims.

```sh
python3 src/kmc-julia/fetch_dataset.py --output /tmp/gisette_scale.bz2
PYTHONPATH=/tmp/pasc-julia-plot-tools python3 src/kmc-julia/cold_regenerate.py --dataset /tmp/gisette_scale.bz2 --materialize
python3 src/kmc-julia/prepare_cold_config.py --output /tmp/kmc-fresh.json
PYTHONPATH=/tmp/pasc-julia-plot-tools python3 tools/verify_portable_julia.py --config /tmp/kmc-fresh.json
```

The cold helper checks exact asset hashes, materializes only missing CPU binary files, and recreates child runtime-asset symlinks. The config helper relocates source paths and removes historical reuse for fresh four-model measurement. Use pinned projects/native toolkit from `environment_snapshot.json`. The verifier locks GPU6 internally: do not add an outer flock. `audit_saved_outputs.py` independently scans all saved values/all30 pair-trials, exact fields, original-source final-case equivalence and39 bound source/runtime hashes. GPU captures and original-source raw diagnostics remain local, with complete hash/path provenance in the source packet.

Actual prospective translation input/cache/output/reasoning counters and configured-model metadata are in canonical STATE manifests/calibration and `observed_token_windows.json`. CUDA.jl creates the common host workflow, KA then JACC reuse it. Counts include inherited context, cached input and coordination; cached/reasoning are subsets. They are neither exclusive billed cost nor recovered historical consumption. Native preparation, verification and annotation have separate windows. No missing historical cost is reported as zero.
