# Gerbil: active GPU kmer-counting component

This study ports the source's active phase-two GPU cooperative hash insertion and complete host table extraction, count-frequency histogram, and threshold selection. It is not the full CLI: compressed sequence readers, minimizer/temp-file partitioning, the mixed CPU/GPU distributor, output writer threads, and emergency overflow evacuation remain unverified.

The active source comments out `compressEntries`; that dormant compaction helper is neither timed nor claimed ported. Native insertion is unchanged. No observed native GPU failure or universal framework limitation is claimed.

## Scientific contract

Five synthetic canonical DNA workloads use k=8,28,60,92,135, with 1/2/4/6/10 UInt32 words per key and 33/257/4099/4099/30000 input keys. Original source KMer packing and reverse-complement normalization were checked for every generated family. Independent Python arbitrary-width packing and Counter establish all per-key counts, all 512 frequency bins, threshold-selected rows, and total/unique/below-threshold statistics. The k135 case also exercises the >=512 count bin and source stream rotation 0,1,2,3,0 over five 256KiB bundle calls.

These capacities require zero overflow: any nonzero failure count rejects the configuration. Vendor overflow sorting is unmeasured, not unsupported. Synthetic component checks do not establish assembly or variant-call quality on real genomes.

## Complete numerical and physical-state evidence

Every accepted trial records 1,176,755 UInt64 values, including a 756,039-value exact input prefix. Initial table/failure storage/counter are read back. Immediately after each HtoD upload, a same-stream DtoD echo captures every actual input word before that stream buffer is consumed by the insertion kernel. This added audit copy is inside every model's ROI.

All physical hash rows and all four stream buffers, failure storage, echoes, counters, and guards are retained separately. Complete canonicalization sorts occupied rows by the source UInt32 key comparator and retains unchanged zero rows. Its full permutation is proved bijective; duplicate keys, locked counters, nonzero discarded payload, lost words, overflow, and guard changes fail. No physical slot identity across concurrent implementations is claimed (`raw_output_identity_claim:false`). This is complete keyed numerical agreement, not checksum or property-only evidence.

`audit_raw_counting.py` independently verifies raw buffers, the full permutation, and each corresponding primary artifact. `raw_counting_audit_all4.json` binds 100 raw case files and their permutation files: 41,347,180 raw words across all 20 accepted model trials. Scientific output arrays and all guard bits are integer-exact; no pointer or integer passes through Float64.

## Framework mapping and timing

Each target has one authored entrypoint, five exercised width specializations, and nine timed launches per trial. CUDA.jl, KA, and JACC passed their first complete device candidates and first GPU/JIT processes. KA owns local-memory allocation; CUDA.jl and JACC reserve static shared scratch through CUDA.jl. All three are explicitly `hybrid_device_intrinsics`: custom LLVM volatile helpers, adapted from the audited BH port, preserve source polling; CUDA warp synchronization, global atomics, and device fences remain backend extensions. Exact reuse provenance is recorded.

Targets add explicit 32-lane barriers between cooperative phases. This is a disclosed synchronization adaptation; native source passes without them. It is not a native repair or evidence of a source synchronization defect. No cross-backend portability claim is made.

Five checked trials follow full warmup; report mean trials 2–5. The preallocated ROI includes source bundle uploads, audit echoes, cooperative insertion, whole-context completion, complete table/counter transfer, and source-policy host extraction/histogram/threshold selection. Allocations/clears, prefix reads, canonicalization, raw capture, and independent checking are outside. Payloads retain 256-byte alignment, with immediate logical guards. No allocation-inclusive full CLI or bare-kernel timing claim is made. Added barriers, host control, and audit copies confound causal abstraction explanations; one ordered block per model does not establish a stable ranking.

## Generate fixtures, rebuild checker, and replay

From the repository root:

```sh
python3 src/gerbil-julia/generate_probe.py
python3 src/gerbil-julia/generate_fixture.py
g++ -O3 -std=c++17 -shared -fPIC '-DGERBIL_ROOT="/home/imo/HeCBench_julia/src/gerbil-julia"' src/gerbil-julia/study_check.cpp -o src/gerbil-julia/study_check.so
python3 tools/verify_portable_julia.py --config src/gerbil-julia/study_config_native.json
python3 tools/verify_portable_julia.py --config src/gerbil-julia/study_config_cuda_jl.json
python3 tools/verify_portable_julia.py --config src/gerbil-julia/study_config_ka.json
python3 tools/verify_portable_julia.py --config src/gerbil-julia/study_config_jacc.json
python3 src/gerbil-julia/audit_raw_counting.py --result PATH_TO_FRESH_RESULT_JSON --output fresh_raw_audit.json
```

`asset_provenance.json` binds the essential generated inputs: `fixtures.bin` (1,858,719 bytes, SHA256 `2a5852fce0b00230ceed0f5262ca88b5326d65392cbe27986823e4b2bb2a94a9`) and historical readiness `probe.bin` (2,508 bytes). `study_check.so` is built locally and not committed. C++17, CUDA 12.0/sm70, Julia 1.12.6, CUDA.jl 6.2.1, KA 0.9.45, and JACC 1.4.0 were used.

Replace absolute root/config paths and the persistent raw-output directory (the final program argument) on relocation. Fresh replay must remove historical reuse fields or point to new matching captures. The runner owns GPU6 serialization; do not wrap it in another lock. Raw trial/slot artifacts stay in local evidence; standalone receipts retain their hashes. Original source and license are archived under `original_native/`.

The first diagnostic probe passed but its temporary binary was not retained because it emitted no trial rows. The unchanged-kernel second probe writes persistent output; `native_probe_audit.json` and `audit_native_probe.py` check every count and guard. A separate CPU-only audit typo is recorded as shared tooling failure. Neither becomes a framework or native mathematical failure. Prospective target token windows, native preparation, and annotation are separate, context-inclusive activity counts; cached/reasoning fields are subsets, not additional or billed cost.
