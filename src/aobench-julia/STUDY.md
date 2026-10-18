# Aobench: ambient-occlusion renderer

## Contract and scientific scope

The original 256×256 scene contains three Float32 spheres and an infinite plane. Each pixel uses 2×2 subpixels and 8×8 cosine-weighted hemisphere samples for each primary hit. The source UInt32 xorshift stream is seeded by `y*256+x` and reset on every launch. Repeating the render 100 times does not add independent Monte Carlo samples.

An independent NumPy 2.5.4 Float64 oracle uses analytic ray quadratics, plane distances, geometric normals, an orthonormal frame and true π. It imports no C++/Julia geometry code or GPU outputs. Scene and epsilon values are widened from their source Float32 bits. The oracle also checks positive hit distances and unit normals.

Before GPU comparison, the contract declared a maximum difference of **one grayscale code** across all 196,608 RGB bytes. This is approximately one visibility vote among the 256 subpixel/sample combinations for a fully hit pixel. It is a finite-precision and quantization acceptance budget, not a statistical confidence interval. Scene/parameter bits, guard bytes, equal RGB channels and the black sky row must agree exactly. Arbitrary scenes, other resolutions, exceptional values, perceptual quality and PRNG statistical quality are not certified.

All 20 captures pass. Native differs from the Float64 oracle at three grayscale pixels (nine RGB bytes), each by one level. CUDA.jl, KA and JACC match the oracle and each other exactly. Each local artifact has 196,828 bytes: a 92-byte exact input prefix, the complete RGB image and 128 exact guard bytes. Native scene-buffer readback and its by-value plane match Julia's source-bound literal constants; Julia is not dynamically parameterized by that buffer.

## Reference defect and repair

Historical CUDA.jl contained a genuine renderer but no numerical check. Its checked `Int32(vv*255.5f0)` conversion threw on fractional grayscale values. Normal and debug GPU executions failed; the debug stack identifies `Int32 → clamp_byte:83 → render_kernel!:193`. A CPU witness confirms that `Int32(127.75f0)` throws, whereas the native C cast produces 127.

The one-line change to `trunc(Int32, ...)` matches native truncation. Original files, failed candidates, logs and before/after hashes are retained. Native CUDA, RNG, scene, arithmetic policy and the predeclared tolerance are unchanged. This is a CUDA.jl runtime repair, not a native correction or an unsupported KA/JACC feature.

## Mapping and measurement

All models use 16×16 threads, 16×16 blocks and one pixel per worker. Ray and RNG state are private; no cooperative shared memory or barriers are used. CUDA.jl uses hardware pixel indices, KA uses Global NTuple indices and JACC uses its two-dimensional parallel loop with explicit zero dynamic shared memory.

Native uses a global sphere buffer, a by-value plane and its original Makefile `-O3 -use_fast_math` policy. Julia uses literal scene values and ordinary Float32 square root, trigonometry and division. These differences are performance covariates. No instruction, occupancy, roofline or bottleneck explanation is inferred without profiles.

A complete render warms the implementation before five synchronized loops of 100 renders. Times are per render; means use trials 2–5. Allocation, transfers, reset, readback and serialization are outside the timed region. Resident buffers are reused; the original program allocates outside each render timer, so memory lifecycle and cache state remain qualifications.

| Model | Mean per render, µs |
|---|---:|
| CUDA | 127.938 |
| CUDA.jl | 1396.080 |
| KA | 1397.300 |
| JACC | 1402.976 |

## Effort and reproduction

CUDA.jl required two source candidates, three actual device checks (normal failure, debug failure, repaired pass), one runtime feedback repair and one generic device kernel. Debug mode separately precompiled 53 packages in 141 seconds; those are not device JIT attempts. Native built once and its five-trial series was reused unchanged. KA followed by JACC each passed one candidate and one device check without repair. Shared pixel helpers, the oracle and reference work are counted separately. Internal compiler passes and historical translation consumption remain unknown.

Prospective phase windows include repeated/cached context and coordination. The broad workflow overlaps its child windows and is not additive or exclusive billed consumption. Source proxies remain separate and partial. Root owns commits and paper updates.

Reproduce with `python3 tools/verify_portable_julia.py --config src/aobench-julia/study_config.json`; the runner serializes GPU6. Regenerate CPU fixtures with `OPENBLAS_NUM_THREADS=1 PYTHONPATH=/tmp/pasc-julia-plot-tools python3 src/aobench-julia/generate_geometry_oracle.py`. See `study_contract.json`, `fixture_provenance.json`, `reference_cast_repair.json`, `saved_output_audit.json` and `source_integrity_review.json` for exact provenance and boundaries.
