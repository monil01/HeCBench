# doh: source box-Hessian response with a genuine existing Julia kernel

The unchanged native kernel and genuine historical CUDA.jl kernel are compared on identical frozen integral images. The original Julia source/Makefile are archived; its only source edit is an import guard around the final main call. The actual kernel and integral helper body are byte-identical. Historical translation attempts/cost remain unknown; this is an existing-reference validation, not a newly authored CUDA.jl GPU translation.

Four image fixtures and four Float32 sigma values yield16 parameter paths: nonnegative texture, positive impulses/boundaries, signed operator control and a single-row boundary control; sigma1/3/4/4.25 maps to integer sizes3/9/12/12. The source explicitly warns that sigma<3 is a poor approximation to a true Hessian. Signed/single-row/small-scale controls certify the discrete operator, not Gaussian derivative or image-feature quality. This is not a complete SURF detector or default CLI runtime claim.

## Numerical contract

The independent CPU oracle directly sums raw pixels rather than copying integral-image corner lookups. Selected raw/integral images are bounded integers:4sum(abs(raw))<2^24 ensures original naive Float32 integration, Julia prefix integration and corner integer operations are exact for this selected class. General Gaussian RNG/integral-rounding equivalence is not claimed.

Source boundary policy is retained: clip the starting coordinate before adding rectangle extent, use inclusive integral corners representing raw rows(r,r2]/columns(c,c2], then clamp a negative box sum to zero. This excludes the first raw row/column. It is not ideal zero padding or an independently validated continuous Hessian.

Each trial retains14074 Float64 artifact values:2722 exact widened input/parameter values plus all5420 actual Float32 output pixels and guarded device input echoes. Output widening is lossless and does not mean Float64 GPU computation. All512 guards/input echoes are exact; responses use predeclared atol1e-4+rtol2e-6 against both GPU references and the raw-pixel oracle. All corresponding sigma4/4.25 outputs have identical Float32 bits, and single-row responses are zero. Original CUDA.jl uses left-associated0.81*dxy*dxy while native uses0.81*(dxy*dxy); both are retained and tolerated drift is recorded, without an isolated code-generation causality claim.

## Porting and timing

KA authors the shared target helper during its translation phase; JACC subsequently reuses it through a genuine JACC kernel. This authorship belongs to the observed KA phase, while the source proxy is stored once in shared accounting. Reference preparation did not author that target helper. This asymmetry and prior source reuse preclude interpreting token windows as intrinsic cold framework costs. Every current reference/target check passed first; no failure or repair is fabricated.

Five checked outer trials each sum16 host-inclusive100-launch batch regions divided100. Only trial1 timing is discarded; per-path values are retained. Input construction, allocation/copies, resets, JIT warmup, readback/oracles/serialization are outside. All models use256-thread ceil grids and tail checks. JACC shared bytes are explicit0; cached LaunchSpec lookup is included in host timing. No profiler supports bandwidth, occupancy, cache or intrinsic framework-overhead attribution.

Use audit_saved_outputs.py for CPU-only full saved-output/input/guard/dependency/timing rechecks, and the central serialized GPU6 runner for GPU work. Root alone commits.
