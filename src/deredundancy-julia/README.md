# Deredundancy: forward nucleotide-read redundancy clustering

The full14-GPU-entry pipeline performs ASCII nucleotide conversion, packed sequence construction, k4/5/6/7 inventories, host compaction/sorting, duplicate merging, composition/word filters, source integer LCS and host-controlled representative clustering. All source GPU bodies and geometries are preserved in a conditional corrected native secondary.

## Source correction and evidence

Original immutable source and license are retained under original_native/. Original host sorting includes leading invalid65535 windows in its valid-word prefix, dropping valid tail windows. Source-extracted CPU bodies and independent Python inventories establish the defect; the original GPU full inventory check fails with intact inputs and guards. study_corrected_main.cu compacts all non65535 windows before the existing prefix sort; original14CUDA GPU bodies remain unchanged. reference_provenance.json binds original/corrected source and witnesses. Original and corrected final clusters/FASTA happen to match on all four fixtures: no observed wrong biological-group claim. The corrected secondary is excluded from untouched-native aggregates.

The first CUDAjl program had a shared host-comprehension parser whitespace error, then a distinct packing error from Julia/C shift-multiply precedence. Exact source archives, CPU AST/bit-packing witness and full failed output artifacts preserve both; explicit parentheses repair the shifted bit count, with no tolerance/input change. Neither issue establishes unsupported private arrays: all14GPU kernels, including private Int16[3000] rows, compile and pass.

## Full contract and scope

Cases n17/17/33/129 use lengths128..192, k4..7 and thresholds.81/.86/.91/.96. Duplicates, mismatches, fragments, half-swapped negatives, homopolymers, lowercase and N are exercised; U/u is source-reviewed but unmeasured. No empty/allgap, zero-valid-word, long-read, reverse-complement, real missing FASTA dataset, or exhaustive biological recall claim. Source strict heuristic filters and ceil(original query length*threshold) LCS acceptance are preserved. The named private-row traversal is bounded; no general all-length safety claim.

Every trial records2,903,055 Int64 values including380,585 exact actual-device-input values. All final arrays and complete FASTA bytes are checked, then a separately tagged untimed deterministic replay captures raw kmer windows, all65536-entry active/cleared tables and per-stage cluster states; replay final must equal the actual timed final. Replay is not falsely described as timed intermediate capture. All13 GPU buffers have both-end guards. Independent Python full-integer LCS, valid-kmer inventory, source filter/representative policy and FASTA output check all values. Raw GPU trial arrays are uncompressed; only the generated fixture's expected oracle record uses lossless zlib.

Five checked trials follow full pipeline warmup; report mean2–5. Each trial sums four preallocated full-pipeline ROIs, including initial input transfers, all120 GPU launches, CPU compaction/sort and host representative transfers/control/finalcluster copy with whole-context completion. Allocations, reset/input capture, full final capture/FASTA printing, tagged replay and checking are outside. Original allocation-inclusive CLI is unmeasured.

CUDAjl uses its exported static shared/barrier APIs. KA uses own local memory/barrier. JACC owns dispatch/barrier, but faithfully reserves the filter's512-byte typed scratch with CUDA.CuStaticSharedArray, explicitly hybrid_device_intrinsics. Dynamic shared size0 and exact source geometry are explicit. Declared LCS private payload12136bytes/thread does not establish actual local-memory spills/occupancy; no profiler counters or causal bottleneck claim.

## Rebuild assets/checker and replay

From repository root (replace absolute root macro/paths when relocating):

```sh
python3 src/deredundancy-julia/generate_fixture.py
g++ -O3 -std=c++17 -shared -fPIC '-DDERED_ROOT="/home/imo/HeCBench_julia/src/deredundancy-julia"' src/deredundancy-julia/study_check.cpp -lz -o src/deredundancy-julia/study_check.so
python3 tools/verify_portable_julia.py --config src/deredundancy-julia/study_config_native_corrected.json
python3 tools/verify_portable_julia.py --config src/deredundancy-julia/study_config_cuda_jl.json
python3 tools/verify_portable_julia.py --config src/deredundancy-julia/study_config_ka.json
python3 tools/verify_portable_julia.py --config src/deredundancy-julia/study_config_jacc.json
```

fixtures.bin is the sole required generated binary input (84,698bytes); asset_provenance.json records SHA/generator/provenance. study_check.so is a locally generated checker, not a committed binary. C++17, zlib development headers/library, nvcc12.0/sm70, Julia1.12.6, CUDA.jl6.2.1, KA0.9.45 and JACC1.4.0 were used. Missing/truncated assets fail. All declared dependencies and input bytes are hash-bound. The runner owns GPU6 serialization; never outer-lock it. Fresh replay must remove historical reuse fields or point to newly generated matching runs. The original-native config intentionally challenges the valid-word scientific inventory and must not be promoted. Context-inclusive actual target usage and disjoint scientific preparation, source/JIT/process counts and RQ1–5 observations accompany each version; unknown historical/backend invocation counts remain null.
