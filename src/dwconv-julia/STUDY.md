# Depthwise convolution: complete source parameter sweep

## Operator and numerical evidence

The study computes per-channel spatial convolution with depth multipliers 1–4, square filters 1, 3, and 5, per-output-channel bias, and zero padding. All **12 source CLI cases** are included. The named fixture uses batch 2, input channels 3, height 11, width 13, unit stride/dilation, padding 1, and ten repetitions. The 1×1 case has a bias-only outer output ring, so border behavior is checked rather than omitted.

Independent NumPy `sliding_window_view` and Int64 `einsum` evaluate signed inputs, weights, and biases using integer arithmetic scaled by 1/256. Products and partial sums are exactly representable in Float32 (bounded scaled magnitude below 3328), which justifies predeclared zero tolerance. This fixture checks signed cancellation, channel/depth indexing, padding and bias; it does not measure ordinary nonrepresentable rounding or trained-network accuracy.

Every one of the 20 accepted artifacts contains 11,652 prefix values, 26,220 complete outputs, and 1,536 guards. The prefix includes actual GPU input/weight/bias readbacks plus lossless geometry/configuration descriptors, not a raw capture of the complete GPU argument ABI. All input-prefix bytes, outputs, and guards match exactly against both references and the independent oracle. Outputs are finite. `max_rel_error=0` follows from verified full finite exact equality, including zero outputs, and is not an uncomputed placeholder. `saved_output_audit.json` independently recomputes these checks and validates all current bound source/fixture bytes.

## Implementations and fidelity

The native kernels are unchanged. The genuine historical CUDA.jl kernel is copied byte-for-byte into the reference child; a separate driver supplies fixtures and repeated measurements. KA uses GlobalLinear indexing; JACC uses parallel_for with explicit zero dynamic shared memory. Each Julia implementation has one generic device-kernel definition and one worker per output, with 256 threads per block. The complete serial spatial dot product and flattened NCHW indexing are retained. All three are `framework_device_kernel` implementations.

Native 1×1/3×3 kernels use compile-time template specialization and pragma unrolling, whereas the historical and portable Julia kernels retain dynamic loops and flat-array arguments. Native accessor ABI, specialization, and code generation are performance covariates. Consequently timings do not isolate a pure frontend effect. No vendor-library substitution, native reference repair, end-to-end model-quality claim, or bandwidth/roofline attribution is made.

## Compiler loop and packaging correction

Reference instrumentation passed in its first evaluation, with no arithmetic repair. KA and JACC each used one authored device candidate. The first KA evaluation passed all numerical checks but was launched from an accidentally lowercase child folder. Its code, configuration, result hashes and samples are preserved under `dwconv-julia-KA/candidate_history/lowercase_path_attempt01`. A coordinator-requested source-path correction copied the identical body into the canonical folder and measured KA again. The old bundles are explicitly excluded from primary selection for packaging before the rerun, never because of speed. Thus KA has one authored device candidate, two process evaluations and one packaging/provenance correction; no compiler-feedback or numerical repair.

The coordinator correction arrived just after JACC's first evaluation launched. That immutable JACC run completed successfully and its sole series was reused, together with unchanged native/CUDA.jl series, when measuring canonical KA. The runner then recomputed all five dual-reference comparisons. This ordering deviation is recorded in `packaging_path_review.json`; it is not an additional JACC attempt. Twelve geometry cases, native template variants, timed repetitions and trials are not translation attempts. Internal compiler/cache jobs remain unknown.

## Timing and scope

Each case is warmed before timing, then executes ten launches with whole-device completion. Allocation, copies and checks are outside the ROI. A trial reports the sum of twelve per-case launch-loop means. Five checked in-process outer trials are retained; the reported mean excludes trial 1 and averages trials 2–5. Pinned Julia 1.12.6, CUDA.jl 6.2.1, KA 0.9.45, JACC 1.4.0, local CUDA 12.0, V100 sm70.

Mean summed kernel time, microseconds:

| Native CUDA | CUDA.jl | KA | JACC |
|---:|---:|---:|---:|
|55.913725 | 70.842775 | 84.880025 | 113.154275|

These are small/cache-resident cases. Original batch 128, channels 16, 224×224/R100 production size, disabled bias, alternative strides/dilations and nonsquare filters are unmeasured. Historical translation effort is unknown. Current shared reference/oracle preparation, target translations, packaging correction, and annotations have separate prospective usage windows; cumulative context/cache/coordination-inclusive counters and saved-source proxies are not exclusive billed tokens.

Final evidence: `.porting-state/portable_julia/runs/20261018T061625382479Z_dwconv/result.json`; executable contract: `study_config.json`. Essential CPU-generated fixtures are `study_case01_f32.bin`–`study_case12_f32.bin` and `oracle_case01_f32.bin`–`oracle_case12_f32.bin`, with generator/hash provenance. GPU captures remain local evidence. Coordinator owns separate verified-variant commits and paper integration.
