# LZSS: full source codec on real QCLOUD bits

## Certified scope

Configuration `lzss_QCLOUD50MB_511_512_513_unique_zeros_fullcodec_exactbits_v1` runs the complete original three-kernel codec on the exact 50,000,000-byte author-supplied QCLOUD file, plus five controls: 511/512/513-word boundary files with IEEE bit patterns, 2,048 unique words, and 512 zeros. The real case uses 24,415 blocks of 512 UInt32 symbols, including 480 zero-padding words. Float-looking payloads are uninterpreted bits: NaNs and signed zero require bit identity, not floating-point comparisons.

The upstream input is already quantized. This study certifies no additional bit loss; it does not validate the upstream physical error bound, an optimal compression ratio, or an arbitrary file-format frontend. The source match search, strict tie policy and greedy tokens are retained; no canonical optimal-LZ77 claim is made.

The original source file and its device kernels are unchanged. An actual original CLI run on the full recovered dataset completed without reporting a mismatch. The named adapter adds allocation guards, defined workspace, full capture, independent checks and summed source compression/decompression timings. It is integration-adapted and excluded from untouched-native aggregate rankings, with no mathematical native correction. Source `delete hostArray` after `new[]` is a latent host cleanup mismatch; no actual cleanup failure was observed. Non-word-aligned files violate the selected UInt32-domain precondition and were not run. Exit zero alone is insufficient: the original checker prints the first mismatch but still returns zero.

## Four implementations and timing

Native CUDA invokes the exact source kernels and CUB exclusive scans. CUDA.jl, KA and JACC implement all codec stages and hierarchical GPU exclusive scans using their own indexing, local memory, barriers, allocators and launch APIs. Each target has five kernel definitions, including two global-scan kernels. All three are `framework_device_kernel`; there are no foreign CUDA device calls or data-dependent host control. Shapes alone determine scan recursion.

The source mixed-type shared storage becomes an explicit little-endian byte workspace: 5,188 bytes for match/encode and 2,048 for scans. JACC.shared requests zero-template preloads and a barrier; emitted traffic was not measured. Nonshared JACC launches explicitly request zero shared memory. Native CUB versus authored scans, byte-level word access, scratch allocation/pooling, launch structure and shared preload are performance covariates. The data do not isolate a causal frontend effect or establish a roofline position.

Every model performs an untimed complete warmup followed by five checked trials. Report the mean of trials 2–5. The ROI sums compression and decompression across all six cases. Compression includes local matching, exclusive scans, packing and scratch allocation/free; source CUB scratch allocations remain inside the native ROI. Decompression includes completion. Actual input readbacks, guards, independent checks and artifact writing are outside it. Compression and decompression seconds are retained separately in every raw row. The original printed throughput uses MiB/ms while labeling GB/s; structured seconds are used here.

Final result: `.porting-state/portable_julia/runs/20261019T231942942062Z_lzss/result.json`. Means: CUDA 3.830385 ms; CUDA.jl 4.851716 ms; KA 5.052416 ms; JACC 5.949182 ms. These are named guarded-pipeline measurements, not a pure compiler or kernel-only comparison.

## Numerical and scientific evidence

Each of 20 actual trial captures contains 156,165,962 bytes: 50,020,544 bytes of exact actual device-input/guard prefix, followed by all decoded padded words, both block-size arrays, both exclusive-offset arrays, all initialized temporary streams, all live packed flag/data bytes, and allocation guards. Uninitialized/unused packed tails are not interpreted as outputs. All 25 required native/CUDA.jl reference and target-versus-both-reference comparisons are byte exact. The complete CPU-control comparison of all 20 captures additionally establishes KA/JACC pair identity.

A scalar CPU encoder reproduces source token choices. A separately structured streaming decoder checks every token bound, distance/length, complete byte consumption and every padded decoded bit against input. It is an independent losslessness check; the scalar encoder alone is not an independent compression-optimality oracle. `saved_output_audit.json` records full rehash and complete streamed comparison of every saved capture (3,123,319,240 bytes), not just hashes. `cold_regeneration_audit.json` and `source_bindings_audit.json` retain exact cold fixture and measured-dependency proofs. No extra GPU timing series were generated for these audits.

## Cold reproduction

Use unprivileged existing tools only. Python standard library and g++ are sufficient for fixtures:

```sh
python3 src/lzss-julia/regenerate_fixtures.py --output src/lzss-julia
python3 src/lzss-julia/prepare_cold_config.py --output /tmp/lzss-fresh.json
PYTHONPATH=/tmp/pasc-julia-plot-tools python3 tools/verify_portable_julia.py lzss --config /tmp/lzss-fresh.json
```

The first command downloads only the pinned upstream commit object if needed, enforces byte size and SHA256, builds the CPU generator, and requires all generated assets to match committed hashes. Pass `--dataset /path/QCLOUDf48.bin.f32.errctrl.1e-3` to reuse the exact external file. The fresh-config helper relocates source paths and removes historical result reuse. The central verifier acquires the GPU6 study lock internally: do not wrap it in another flock. Match pinned project manifests/local toolkit preferences and inspect `environment_snapshot.json`. Runtime fixtures and all includes are dependency-bound before measurement.

Large CPU-regenerable binaries and raw GPU captures stay local; source Git needs recipes, upstream identity, hashes and proofs, not those bytes. All four generated assets reproduce exactly on the recorded little-endian Linux/g++ environment.

## Effort and usage

CUDA.jl: two complete-program evaluations, one authored-device JIT process, one host-transfer API repair. The first program's ReinterpretArray/SubArray H2D copy fell back to forbidden scalar indexing before authored kernels; a materialized host Vector fixed it. KA: one program/JIT process, zero repairs. JACC: one actual program/JIT process, zero kernel repairs; a Python generator syntax error and consequent missing-dependency verifier rejection occurred before Julia/GPU and are separate setup effort. Each target submits one five-kernel source bundle; KA/JACC reuse reviewed mathematical bodies and common harness, not independent clean-room synthesis.

Actual context-inclusive translation counters: CUDA.jl 1,062,775; KA 795,732; JACC 1,290,370 total tokens. Cached input and reasoning are subsets. CUDA.jl END shared a tool round with final phase-manifest writing; preserve that scope undercoverage and exclude its calibration record. Separate KA/JACC END rounds followed metadata. Native preparation, saved verification and annotation are separate windows, with original stream/model metadata and marker lag retained. These are neither exclusive billing nor recovered historical counts. Annotation had one Python metadata-script whitespace syntax error, corrected without touching measured source or GPU evidence; it is not a target compiler failure.
