# dxtc2: cooperative BC1/DXT1 block compression

The genuine historical Julia compressor is preserved as original_main.jl with its Makefile. Its serial covariance/search ordering and half-up endpoint rounding differ from the native64-lane cooperative implementation. New CUDA.jl/KA/JACC kernels restore the native covariance tree, strided992four-color/160three-color candidate search, lane winner reduction and nearest-even RGB565 endpoint quantization. This is source algorithm restoration, not a claim the old compressor always fails or that its historical creation cost is known.

## Numerical and quality contract

Twelve nondegenerate affine RGB-line4x4tiles with permuted pixel order and alternating ignored alpha are compressed once in full, then a5-block subset beginning at offset2. All3797 raw UInt32 artifact words are retained pertrial:1221exact actual input/permutation/launchparameter words plus full guarded outputs/table/image echoes for both paths. All34active compressed words agree exactly against both references, all96 physical guards and14 untouched words outside the subrange are exact. No output masking or canonicalization is used.

A separate independent integer BC1 decoder reconstructs every selected tile RGB component and checks source-pixel mean squared error<=256, declared before the first trial. Actual maximum is74.25. This fixture-specific lossy quality property does not prove globally optimal quantization or perceptual quality. Original native CLI compares decoded compressed reference-image differences against0.02, which is a different quantity; full Lena/teapot/DDS reference CLI and degenerate zero-covariance tiles are not certified. Actual floating computation is Float32; artifact encoding is exact UInt32 packed words. Both original native and historical Julia permutation generators match all1024 frozen entries.

## Mapping and measurement

All four run64lanes/tile. Julia versions use an explicit467Float32 shared workspace(1868B), bitcasting permutation UInt32 bits without arithmetic and storing bounded small integer ranks/indices exactly. An extra explicit shared color read-before-scatter barrier ensures a snapshot of all16 colors; no observed native GPU race is claimed. JACC.shared copies the whole467element initialized template and adds a barrier inside its measured kernel; other models do not require that initialization. Dynamic shared bytes are explicitly1868. Exact register usage/cache/spills/occupancy are unprofiled.

Five checked outer trials each sum two host-inclusive10-launch batches divided10; only trial1 timing is discarded. Input construction, copies, reset, compilation, readback, independent decoding and serialization are outside. Cached JACC LaunchSpec lookup is inside the host region. This small tile study is not full image/file walltime or a roofline measurement.

## Failure and effort records

The first owned native measurement harness failed to compile because of a range-initializer/iostream adapter error. The first two Julia captures reported a wrong quality failure because a nested host decoder variable captured its outer block index; raw GPU compressed words already matched native exactly. Both diagnostic series and candidate source states are retained. The host decoder was fixed without changing GPU code or thresholds. New reference GPUcandidate1 was checked in three fresh programs, zero GPU kernel repair; each KA/JACC target candidate passed its first check. These owned adapter T1 events are separate from framework translation failures.

Shared Float32 helpers and the cooperative body template were authored during reference restoration. Target main files mechanically bind that body to each framework and are hashed in full before measurement. Token windows include repeated/cached context, same-benchmark annotations and coordination, overlap with outer workflow and must not be summed or interpreted as intrinsic cold framework costs. Historical creation effort is unknown.

Source archives, scientific_witness.json, output_audit.json and raw diagnostic bundles preserve the contract and limitations. Use audit_saved_outputs.py for CPU-only saved-evidence checks; root alone commits.
