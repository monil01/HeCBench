# CCS: conditional gene-expression biclustering

This study translates the complete GPU conditional-Pearson, greedy augmentation, and strict-largest candidate-selection algorithm, and independently checks the complete source-ordered CPU merge and printed index outputs. Three planted expression matrices are 12×32, 17×64, and 33×200, with 4/5/7 base genes and three internal kernel repeats. The source mapping is one GPU thread per base-gene block.

## Explicit corrected native secondary

The original source is unchanged and archived under `original_native/`. An exact address trace shows three condition masks written with stride 3 but read with stride 200. The original GPU/output pipeline fails full independent checking with intact inputs/guards; its full failed binary remains in local diagnostic evidence. It also copies temporary memberships while returning selected-best scores/counts. `study_corrected_main.cu` changes only nine mask write indices to the existing MAXSAMPLE stride and two host copy pointers from temporary to selected buffers. `reference_provenance.json` binds original/corrected source and independent witnesses. The study is excluded from untouched-native aggregates.

Unused output state is explicitly initialized to zero outside the ROI and captured as actual input. No input equals its supplied gene mean; the source's unassigned zero-deviation classification edge is unmeasured. Degenerate variances retain IEEE NaN-as-false threshold comparisons. D≤200 matches the static source array limit.

## Complete checks and timing

Every accepted trial records 20,226 Float64-encoded values (lossless Float32/UInt8/Int32 expansion), including a 12,606-value exact actual-device-input prefix. Complete selected/temporary masks, counts/scores, merged states, printed index bytes, read-only genes, and all eight buffer guards are checked. Independent Float64 Pearson and complete augmentation/strict-selection/merge/printing replay has threshold margins above 0.064. Intermediate Float32 correlations are not separately emitted; there is no universal correlation-error or biological discovery certificate.

Five checked trials follow complete warmup; report the mean of trials 2–5. Each trial sums the three original kernel-repeat means with explicit context completion. Allocation/upload/readback/checks and CPU merge/printing are outside this kernel ROI. The source input reader/variance ordering, missing original dataset/reference-output files, and full allocation-inclusive CLI timing are unmeasured.

CUDAjl uses its exported static shared API; KA uses its own local-memory arrays, with no CUDA device extension. JACC owns dispatch but reserves three faithful typed static arrays through CUDA.jl, explicitly `hybrid_device_intrinsics`. Its supported `shared(vector)` API copies global input and synchronizes, which would add work relative to source scratch reservation; this is a documented mapping choice, not an unsupported algorithm claim.

## Generate assets and rebuild the local checker

From the repository root:

```sh
python3 src/ccs-julia/generate_fixture.py
g++ -O3 -std=c++17 -shared -fPIC '-DCCS_ROOT="/home/imo/HeCBench_julia/src/ccs-julia"' src/ccs-julia/study_check.cpp -o src/ccs-julia/study_check.so
```

`asset_provenance.json` records the only required binary input, `fixtures.bin` (93,596 bytes), its SHA256 and generator. It is synthetic, not the missing real expression dataset. `study_check.so` is generated locally, not a source asset. Replace the root macro and absolute config paths when relocating. All declared dependencies and runtime fixture bytes are bound in the verifier; missing/truncated files fail.

## Replay

```sh
python3 tools/verify_portable_julia.py --config src/ccs-julia/study_config_native_corrected.json
python3 tools/verify_portable_julia.py --config src/ccs-julia/study_config_cuda_jl.json
python3 tools/verify_portable_julia.py --config src/ccs-julia/study_config_ka.json
python3 tools/verify_portable_julia.py --config src/ccs-julia/study_config_jacc.json
```

The runner serializes GPU6 itself; do not wrap it in another lock. Fresh replay must remove historical reuse fields or point them to newly generated matching captures. Recorded versions: Julia1.12.6, CUDA.jl6.2.1, KA0.9.45, JACC1.4.0, native CUDA12.0/sm70. The original-reference config intentionally fails the independent corrected scientific contract and must not be promoted. Raw trial artifacts stay local; proofs, current effort/token records, source-reviewed ontology and RQ1–5 observations preserve scope and costs.
