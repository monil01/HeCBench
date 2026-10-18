# FP32 key/value paged-cache reshape and E4M3 quantization

This is an attention **storage/layout operator** experiment. It does not run attention, generate tokens, or establish language-model quality. Source input shape is 1024 tokens ×32 heads ×128 features. All eight source FP32 cases are retained: page/block sizes16,32 and key packing widths1,2,4,8. Each input tensor has4,194,304 Float32 values. A source-faithful C++ generator retains MT19937 seed42, alternating normal key/value draws, subsequent slot permutation, and positive maximum/64 scales. Hashes/generator versions are in `fixture_provenance.json`.

## Precision boundary

The source also requests eight **BF16-input** cases. Its own `utils.h` conversion specialization asserts for device architecture belowSM80. The minimal unchanged-helper reproducer in `diagnostics/bf16_sm70_guard_probe.cu` compiled forSM70 and actually raised CUDA710/assert on V100; source/log hashes and output are retained in `diagnostics/bf16_guard_evidence.json`. These BF16 cases are hardware/source-policy blocked here. Neither their CUDA.jl, KA, nor JACC support is established by that failure. No BF16 arithmetic is replaced with Float32 under a BF16 label. The complete original16case CLI remains uncertified.

FP32 cases really store **E4M3 byte codes**, not Float32 stand-ins. Source CUDA uses `__nv_cvt_float_to_fp8` with SATFINITE/RN-even after FP32 division. Julia uses pure Julia bit manipulation implementing the same finite-input encoding, including signed zero, subnormals, ties-even and saturation. This is software conversion on V100, **not FP8 arithmetic/tensor-core computation**. All current normal-distributed fixture codes are finite and saturation is not exercised. The Julia encoder is scoped to finite source fixtures; source NaN-conversion behavior is neither implemented nor certified.

## Independent numerical evidence

`generate_study_oracles.py` enumerates the127 finite nonnegative E4M3 magnitudes in Double, selects the nearest with ties-even after rounded FP32 division, retains the sign including negative zero, and computes cache layouts using tensor transposes over the inverse slot permutation. It does not call the device bit encoder or copy its indexing loop. Complete key and value byte arrays are compared exactly after each outer trial, against both unchanged native CUDA and genuine CUDA.jl references and the independent oracle.

The artifact UInt32 format carries raw FP32 input bit patterns, the nonnegative Int64 device slots serialized losslessly as UInt32, scale bits, and seven shape/measurement parameters, followed by both complete FP8 code arrays widened losslessly to UInt32. `input_prefix_count=8389641`; full artifact count=16778249. Widening is host serialization only; device slots remain Int64 and cache allocation remains UInt8. All five captures require exact input-prefix and encoded-output agreement. Exact encoded-byte agreement implies equal decoded FP8 values; it does **not** imply lossless FP32 reconstruction.

`quantization_accuracy.json` separately reports actual FP32-to-dequantized error: key maximum absolute error0.1679706573/RMS0.02644168888, value maximum0.1617724299/RMS0.02642614454. It includes actual signed-zero and tie counts. These are operator quantization errors, not task accuracy.

## Mapping and timing

Original native kernel bodies and conversion helpers remain unchanged. One512-thread block handles each of1024 tokens; each lane iterates over head/feature blocks with stride512. Key cache ordering is `[page,head,feature/x,page_slot,x]`; value cache is `[page,head,feature,page_slot]`, in source physical row-major order. The key x4/x8 path uses native aligned float4-to-packed-byte conversion versus Julia NTuple4 loads/stores; emitted vectorization was not profiled. The x1/x2 path is scalar. Value writes remain page-strided. Native CUDA vendor conversion and pure Julia encoder/code generation can differ in performance even when byte-exact. There is no shared-memory allocation, inter-thread synchronization, atomic update, or sort.

The timing region is the complete original cache launch loop plus device-wide completion, divided by1000 calls. All implementations do30 untimed warm launches, then five outer trials; the first outer time is excluded and the last four averaged. Input copies, allocation, output transfer/serialization, and numerical checks are outside the region. Cache buffers are zeroed once before warmup and then fully overwritten; no inter-trial reset is required. Every outer output is saved, but the999 intermediate overwritten outputs per trial are not independently captured. Five outer trials are four retained performance samples, not4000 independent timing samples. No counter-based bottleneck or roofline claim is made.

The historical CUDA.jl marker/canned PASS is preserved under `reference_original/` and replaced with the real operator. Its restoration is separately costed from KA/JACC translation. Historical effort remains unknown. Source CPU `reference.h` has an early return on a negative slot, whereas a CUDA block skips its own token; this discrepancy is outside the present nonnegative permutation contract and is recorded as an unexercised diagnostic, not a failed primary case.

## Effort and reproducibility

Pinned projects are `_julia_ka_env` (CUDA.jl/KA) and `_julia_jacc_env` (JACC), with CUDA6.2.1, KA0.9.45, JACC1.4.0 and local CUDA12.0 on V100SM70. Target frontend macros and actual device-JIT invocations are counted separately. One process per shape compiles the shape's actual kernel before timed trials; internal compiler passes and generated specializations are unknown. Usage boundaries preserve actual context/cache-inclusive consumed counters, separate shared/reference/translation/annotation phases, and nonexclusive coordination; source-size proxies are partial and not billed-token totals. No numerical success is inferred from a marker or source inspection alone.

## Measured results (microseconds per cache call; mean trials2–5)

| Page/pack | Native CUDA | CUDA.jl | KA | JACC |
|---|---:|---:|---:|---:|
| B16_x1 | 239.91 | 301.75 | 229.34 | 303.11 |
| B16_x2 | 281.56 | 281.15 | 272.34 | 281.69 |
| B16_x4 | 238.08 | 289.58 | 287.24 | 290.45 |
| B16_x8 | 322.95 | 393.54 | 400.07 | 393.95 |
| B32_x1 | 335.87 | 405.98 | 329.48 | 406.74 |
| B32_x2 | 403.83 | 365.82 | 401.63 | 366.74 |
| B32_x4 | 330.56 | 395.93 | 415.01 | 395.65 |
| B32_x8 | 363.78 | 389.65 | 446.90 | 388.44 |

All160savedcaptures pass independent code/layout oracle; all32case/model source and executed dependency hashes match. One program candidate/eight actual GPU process checks per Julia version, zero portable repairs. Eight layout configurations are not eight compiler repair iterations. Timing differences remain unattributed to instruction/occupancy/cache causes without profiling. Restart via eight `study_config_B{16,32}_x{1,2,4,8}.json` configurations or `make run-all` within a child. BF16 inventory/proof remains explicit. Root commits each verified variant separately.
