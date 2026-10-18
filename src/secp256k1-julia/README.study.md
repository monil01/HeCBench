# Exact 512-point field arithmetic GPU study

Run `python3 tools/verify_portable_julia.py --config src/secp256k1-julia/study_config.json` from the repository root. The central runner serializes physical GPU 6 and checks all five full outputs against CUDA and CUDA.jl, reusing unchanged, hash-checked captures. `make run` here runs the named CUDA.jl study under the common lock.

The native source performs a complete 512-point mixed-Jacobian addition chain, multiplies all prefix Z coordinates, inverts that product and returns 32 bytes. The new CUDA.jl/KA/JACC implementations perform that complete integer computation on the GPU. The original Julia file computed it with CPU BigInt and timed only a GPU byte copy; the file and Makefile remain in `reference_history`.

Three fixed tables use the original 512 distinct curve points in canonical, reverse and rotate-17 order. Independent Python integers validate the curve states, nonzero denominators and inverse identity. The affine endpoint agrees across orders, but the returned accumulated-Z inverse is order dependent. Full 32 output bytes and 17 guard bytes per case are checked exactly. GPU intermediate trajectories are not captured. There is no signature, network, key-search, cryptographic-security, exceptional-point API or constant-time certification.

Actual device arithmetic uses UInt32 radix-26 limbs and UInt64 products/carries. Outputs are UInt8, widened losslessly to UInt32 only after transfer for a common artifact format. The kernel retains the source one-thread launch. This is integer latency, not a floating-point roofline or saturated throughput experiment.

`field_math.jl` transcribes the original scalar arithmetic, unsigned casts and inversion addition chain. `translate_field_math.py` is a development utility run from the repository root; do not regenerate frozen measured sources. A preserved CPU candidate exposed an unconverted C pointer arrow in byte packing, fixed before the first GPU check. `reference_limb_cpu_audit.jl` is a separate CPU-only BigInt audit, never included by GPU entrypoints.

Essential binaries: `fixture.bin`, `expected_bytes.bin`; hashes and independent oracle records are in `fixture_description.json`. `independent_artifact_audit.py` checks all 20 captured artifacts without GPU execution. Source/license/provenance, all five timings and metadata-only scope adjudications are retained. No existing timing series or tolerance is replaced.
