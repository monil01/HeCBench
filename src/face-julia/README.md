# Face detector: complete synthetic Haar-cascade pipeline

All four versions execute the original CPU detector and the sole GPU rectangle-address preparation kernel: 25 stages, 2,913 weak nodes, a 24×24 training window, Float32 scale factor 1.2, scanning, and rectangle grouping. CUDA.jl has its own Julia CPU pyramid, integral images, cascade, scan, grouping, and drawing. The C++ interface only decodes fixtures and compares complete outputs. KA and JACC inherit the validated Julia host algorithm and own their kernel dispatch. There is no native detector call from Julia.

Three synthetic images (32×32, 48×40, 65×49) exercise acceptance, rejection at stage zero, and mixed decisions. The original trained classifier and Face.pgm are missing; this study does not measure trained face accuracy or recall. Two rectangles contribute numerically. Optional third-rectangle null and non-null addresses are exercised with zero weight. All 2,913 nodes execute wherever the cascade accepts; only the independent oracle memoizes repeated identical synthetic weak rules.

## Complete numerical and physical-pointer evidence

Each accepted trial contains 654,294 Int64 values, including a 268,449-value exact actual host/GPU input prefix and all 768 GPU guard bytes. The contract covers complete resized images, integral and squared-integral images, every rectangle pointer offset, every window code, preliminary and grouped rectangles, and the drawn image. The independent oracle uses direct pixel-patch sums, explicit UInt32 normalization, integer square root, and connected components.

Intermediate arrays come from a tagged, untimed deterministic replay. Actual timed grouped rectangles must equal replay rectangles exactly. The source's 23×23 corner differences with a 576-pixel normalization area and unsigned variance are preserved; this is not a textbook normalization correction.

Separate raw UInt64 sidecars retain each process-local host base, the first original-kernel returned pointer (the first rectangle starts at 0,0, giving an exact device echo of the base), every returned address, and the actual CPU dereferenced value. Every address, range, alignment, roundtrip, and dereference is independently checked. Twenty files contain 12,584,160 address/dereference pairs. `raw_pointer_audit_all4.json` records their paths, byte counts, and SHA256 hashes. Raw pointers never pass through Float64.

Complete scientific outputs and normalized offsets are compared across models. Literal addresses differ across processes: `raw_output_identity_claim` is false. Native CPU code dereferences pointers; Julia uses the corresponding normalized indices, an explicit addressing adaptation.

GPU allocations have 32-byte 0xa5 guards at both ends. Named CPU logical bounds were reviewed; no arbitrary large-image heap safety claim is made. The first native capture passed all five scientific checks with normalized offsets. Coordinator review then requested additional raw-address evidence; the original capture and source are preserved under `reference_history/`, followed by fresh measurements of all four models. This was an evidence enhancement, not a scientific defect or tolerance change.

CUDA.jl had two shared host failures before authored device compilation: generator whitespace and the runtime guard memset's byte-pointer type. Both are archived. One authored device candidate then passed. KA and JACC passed their first candidates; neither uses a CUDA device extension.

## Timing and scope

Five checked trials follow full warmup; report the mean of trials 2–5. The ROI includes the complete allocation-inclusive detector, plus symmetric input-copy, GPU guard, and raw-pointer/dereference audits. Native progress printing is retained. Fixture setup, final capture, deterministic replay, independent checking, and drawing are outside.

Instrumentation, allocation/GC, CPU addressing, and union-find implementation differences are covariates. This is not an uninstrumented CLI or pure kernel-lowering timing comparison. Fifth-trial spikes are retained; no post hoc outlier removal or causal GC/dispatch claim is made.

## Generate, rebuild, and replay

From the repository root:

```sh
python3 src/face-julia/generate_fixture.py
g++ -O3 -std=c++17 -shared -fPIC '-DFACE_ROOT="/home/imo/HeCBench_julia/src/face-julia"' src/face-julia/study_check.cpp -lz -o src/face-julia/study_check.so
python3 tools/verify_portable_julia.py --config src/face-julia/study_config_native.json
python3 tools/verify_portable_julia.py --config src/face-julia/study_config_cuda_jl.json
python3 tools/verify_portable_julia.py --config src/face-julia/study_config_ka.json
python3 tools/verify_portable_julia.py --config src/face-julia/study_config_jacc.json
python3 src/face-julia/audit_raw_pointers.py .porting-state/portable_julia/face_raw_pointer_evidence_v2
```

The sole essential generated binary input is `fixtures.bin` (801,420 bytes), SHA256 `3a2aadfdab1a94d61c9fce311a662cb5fbbebe5f957fe140285c7419d7d40035`. `asset_provenance.json` records its generator and provenance. `study_check.so` is built locally, not committed. C++17, zlib, CUDA 12.0/sm70, Julia 1.12.6, CUDA.jl 6.2.1, KA 0.9.45, and JACC 1.4.0 were used.

Replace absolute macro/config paths and the persistent raw-pointer output directory (the final program argument) when relocating. For a fresh replay, remove historical reuse fields or point them to newly generated matching captures. The runner owns GPU6 serialization; do not wrap it in another lock. Original sources/license and failed candidates are preserved. Raw trial arrays and pointer sidecars remain in local evidence; audit receipts retain hashes.

Actual token windows are context-inclusive; cached/reasoning counters are subsets. CUDA.jl's two target segments are separated by shared evidence instrumentation and excluded from clean one-window calibration. These are not billed, exclusive code-generation, or reconstructed historical costs.
