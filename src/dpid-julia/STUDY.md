# Pixel-importance image downsampling

## Operator and fixtures

The first kernel computes area-weighted patch colors and quantizes them to RGB bytes. The second uses a local Gaussian average of that guidance, then weights source pixels by their color distance raised to `lambda`. A zero total weight returns the local guidance average.

The study covers two images, both producing 16×12 RGB outputs:

| Input | Patch dimensions | Source samples per output |
|---|---:|---:|
| 40×30 | 2.5×2.5 | 9 |
| 128×96 | 8×8 | 64 |

Lambda values 0, 1, and 0.5 exercise both special branches and the general power branch. The source Makefile also uses 0.5. A constant RGB region forces six zero-weight fallback pixels per fixture for positive lambda. Lambda 0 supplies an exact area-average control.

The independent Float64 oracle uses interval intersections, separable Gaussian weights, and Euclidean color distances. The source's common distance normalization cancels in the normalized weighted mean. The oracle does not copy the Float32 accumulation or warp reduction. Its deterministic generator and file hashes are in `generate_oracle.py` and `fixture_provenance.json`.

## Numerical evidence

All 60 saved captures were checked: four models × three lambda settings × five trials. Each contains:

- 40,480 lossless UInt32 representations of actual input bytes and parameter bits;
- 2,304 guidance and final RGB channel values;
- 576 guard byte values.

Input and guard checks require exact serialized bytes. The payload tolerance was declared before measurement: one intensity level, with zero relative tolerance. Guidance is exact in every case. Lambda 0 produces exact final outputs; lambda 1 and 0.5 differ by at most one level from both GPU references and the independent oracle. This is tolerated numerical agreement, not identical raw RGB output.

For lambda 0.5, the first native capture differs from the oracle in 8 and 14 final channels across the two 576-channel images; each Julia version differs in 10 and 11. For lambda 1, these counts are 8 and 12 for native CUDA and 4 and 6 for each Julia version. The audit retains counts for every trial. No claim about the sole cause of these differences is made without unquantized Float32 traces.

`saved_output_audit.json` checks all captures and 183 source/fixture bindings, including path aliases. The source CPU checker prints a warning yet returns zero for excessive differences. The study instead enforces the declared bound on every value; no historical native failure is inferred from that weak exit policy.

## Mapping and performance limits

Native CUDA assigns one 32-lane warp per output and reduces Float32 values with shuffles. Each stage uses 128 threads and a 4×12 block grid. The existing CUDA.jl implementation assigns one serial worker per output. Its two kernels are preserved byte for byte. KA and JACC reuse those audited serial workers, with 128 threads and two blocks per stage.

All Julia versions are `framework_device_kernel` implementations: ordinary array operations and standard Float32 square root/power, with packed RGB byte storage. JACC explicitly requests zero dynamic shared memory. The different reduction order, cooperation pattern, and block counts confound attribution of native-to-Julia performance differences solely to an abstraction.

These are small, cache-resident images with 192 output pixels. The original 8192×8192 image and Makefile's anisotropic 12.8×17.0667 patches remain unmeasured. Other exponents are unverified. The three lambda configurations share images and are not independent datasets. No image-quality, training-accuracy, bandwidth, or roofline conclusion follows from these timings alone.

## Compiler loop and evidence gaps

The reference device code was already genuine. A new reporting harness initially referenced a loop-local `p` outside its scope. Three initially scheduled lambda evaluations reached the authored GPU work, then failed while formatting the first `STUDY_TRIAL` record. None emitted a record: zero accepted and zero rejected rows. Their unreported binaries were lost during temporary-directory cleanup, so their numerical verdicts remain unknown.

One host reporting correction uses `LAMBDA`. Three subsequent evaluations produced the accepted five-trial captures, reusing the original native series without selecting faster samples. Thus current reference preparation has one device-program candidate, two kernel definitions, two host program revisions, six external compile/execution evaluations, and one host repair. There was no device arithmetic, compiler-feedback, semantic, or tolerance repair. The failed host source is archived under `reference_history/host_attempt01`.

KA and JACC each have one program candidate, two kernel definitions, three configuration evaluations, and no repairs. Internal compiler jobs and cache passes are unknown. Tested lambda values are not distinct dtype specializations. Shared oracle preparation and worker extraction are recorded in the reference phase.

Historical effort is unknown. Prospective usage windows include context, cached tokens, and coordination; the broad workflow overlaps the individual phases. Saved-source proxies include reused or extracted content. Neither measure is an exclusive billed translation cost.

## Timing and reproduction

Each model warms the complete two-kernel pipeline before five checked in-process trials. Each fixture runs ten guidance/downsampling pairs per trial, with whole-device completion. Allocation, transfers, and checks are outside the timed region. The headline sums the two fixture means and averages outer trials 2–5. All samples remain available.

| Lambda | CUDA µs | CUDA.jl µs | KA µs | JACC µs |
|---:|---:|---:|---:|---:|
| 0 | 16.656 | 56.209 | 66.780 | 70.338 |
| 1 | 16.371 | 57.931 | 66.832 | 67.187 |
| 0.5 | 17.367 | 69.097 | 81.149 | 82.884 |

The primary lambda-0.5 bundle is `.porting-state/portable_julia/runs/20261018T055325467404Z_dpid/result.json`. `configuration_results.json` and `study_config_lambda_{0,1,half}.json` retain all configurations. `source_integrity_review.json` records measured-file bindings and the preserved reference body.

Essential CPU-generated fixtures are the two `study_input` files, two `oracle_guidance` files, and six `oracle_output` files. Actual GPU captures stay local. `make run` verifies all three configurations using GPU6 under the shared lock. The environment is Julia 1.12.6, CUDA.jl 6.2.1, KA 0.9.45, JACC 1.4.0, local CUDA 12.0, and V100 sm70. The coordinator handles separate commits and paper integration.
