# Verified native-output box-filter contract with known boundary defect

Named fixture256x128,RGBA8,radius10, both original passes. Retains horizontal zero padding, packed quantization between passes, vertical replication and native outputrow0 resetzero. Original native col_kernel omits row0store; original CPU writes it and original verifier excludes radiusborder. Saved native artifact differs on all256 firstrow pixels from an independent unmasked CPU calculation. Full native-output equivalence is verified, not ideal box-filter boundary correctness.

Historical mcpr alias is preserved in boxfilter-julia-cuda.jl/original_main.jl; genuine CUDA.jl reference repair is separately costed. KA/JACC use actual shared-row/sliding-column kernels. Native sources are unchanged; build links existing CPU/PPM support from boxfilter-sycl as the native CMake file specifies. PPMIO is outside the study.

All32768 packed values checked exactly in each of five warmed trials, averaging timing2–5 with ten internal repeats. UInt32 pixels are losslessly widened toInt64 only for artifacts after timing. JACC row scratch360B explicit; template initialization overhead disclosed. See study_config.json and local state manifests for raw evidence, source hashes, taxonomy and precision details.

The unmasked firstrow comparison is diagnostic_only, out of scope for the original interior-only verifier and is_failure:false. The unchanged native original validation contract passes; the genuine historical CUDA.jl alias repair is a separate action. No native reference correction was made.
