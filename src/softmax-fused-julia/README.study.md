# Typed scaled and masked softmax

Run `python3 tools/verify_portable_julia.py --config src/softmax-fused-julia/study_config.json` from the repository root. The central runner locks physical GPU 6 and compares every full capture against CUDA and CUDA.jl; `make run` runs the named Julia reference.

The 16 paths cover actual Float16/BFloat16 storage, Float32 accumulation, both BERT/GPT mask broadcasts and four shapes. Fully masked rows and all masked probabilities are exactly zero. The original source finite sentinel −10000 is retained, with finite unmasked scaled inputs above that sentinel. Vector-four native paths use divisible key lengths128/1024; key1/31 use scalar paths. The original CLI, partial vector tails, unbounded/nonfinite input domains and model-level accuracy are unverified.

The independent Float64 oracle is rounded to output storage before the source1e-3 comparison. Unrounded probability error and storage-resolution row-mass checks are reported separately in independent_artifact_audit.json. Artifacts widen actual16bit values losslessly to Float32 and prepend exact actual inputs/masks/scales/shapes. All20 saved artifacts are audited without another GPU series.

Native source requires an exact float-zero cast to disambiguate CUDA12.0 BF16 constructors. Original sources and hashes remain intact; native_kernel_compat.cuh/native_cpu_reference_compat.h change only `(output_t)0` to `(output_t)0.0f` plus a relative include and exclude the unused CLI. This is compilation-only, not a math correction, but conservatively belongs to the conditional-native-reference-repaired stratum and is excluded from untouched-native aggregates. native_reference_provenance.json records the build failure and chronology.

The original genuine Julia code used Float32 for both formats and indexed BERT masks incorrectly across heads. It is archived with its Makefile. Current reference also adds a statically justified barrier between reading a shared maximum and reusing the scratch buffer; the prior passing series remains preserved. This is not a claimed observed GPU numerical failure.

Native warp/subwarp packing differs from the Julia256thread block-per-row tree. Resource/barrier/vectorization/transcendental-work confounds are explicit in traits and the paper observation. All five timing samples, including retained KA spikes with zero recorded GC time, remain unchanged. No inferred bottleneck or universal abstraction ranking is claimed.

Essential binaries: fixture.bin and expected_outputs.bin; hashes are in fixture_description.json. Keep source/license/provenance, candidate archives, numerical audit, digest proof and metadata-only adjudications together. Usage windows include cached/repeated context and overlap during reference completion; do not sum them or treat them as billed intrinsic framework costs.
