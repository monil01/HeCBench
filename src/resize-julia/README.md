# Unsigned image resizing: conditional corrected native reference

This study preserves the original sources and actual native GPU counterexamples. It certifies agreement with an explicitly corrected reference, not the untouched native implementation. All original failures remain in native_original_witness.json with full raw inputs/outputs and hashes.

## Why a correction is required

The native host assigns in_width/out_width to float after integer division. Its integer bilinear templates also cast fractional interpolation weights to the unsigned pixel type, making every fractional weight zero. On a4×4 linear gradient resized to2×2, the unchanged UInt8 GPU kernel outputs0,20,40,60 instead of the convex bilinear15,35,55,75; all32 channel values disagree. A separate5→3 nearest witness has hostscale1 and outputs1,2,3 rather than1,3,4 with the true ratio. The GPU witness observes UInt8; the generic unsigned cast defect additionally follows directly from source semantics for UInt16/32.

The owned correction changes host scaling to true Float32 division and uses Float32 bilinear weights/pixel arithmetic before final unsigned truncation. The exact patch, original and corrected hashes, justification and witnesses are canonical reference_provenance before the first accepted trial. The nearest native kernel body is unchanged. These timings are excluded from untouched-native performance aggregates.

The native nearest coordinate convention is deliberately retained: round-away((x+.5)*scale), upper-clamped, with no−.5 offset. Identity nearest therefore shifts one pixel; this is a named source convention, not a generic nearest identity claim. Bilinear uses max((x+.5)*scale−.5,0), replicated upper neighbors and final truncation. Historical Julia genuinely used floating bilinear arithmetic and true ratios, but nearest used ties-to-even; that source difference is preserved and corrected without inventing a historical GPU failure.

## Exact numerical domain

There are24 paths:UInt8/16/32 storage, four shapes and nearest/bilinear modes, with16 planar channels and three repeats each. Shapes are8×6→4×3,9×7→6×4,12×14→24×28 and5×3 identity. All ratios are dyadic and exactly representable. Nearest covers the full unsigned ranges, including actual UInt32max output, without converting pixels to Float32. Bilinear inputs are bounded by220 for UInt8 and3520 for UInt16/32. High-range UInt32 bilinear interpolation, non-dyadic scales, unselected coordinate flags, leftover channels and the original large CLI are outside certification.

An independent arbitrary-precision Fraction oracle computes convex four-corner weights rather than the expanded GPU polynomial. Coordinate fractions have denominator at most8 and products at most64; the bounded expanded intermediate numerator is below2^21, within the24-bit Float32 significand. Every arithmetic step is exact in this selected domain before final truncation, so atol=rtol=0 is justified. This does not imply exact interpolation for arbitrary Float32 ratios or high UInt32 intensities.

All20 saved artifacts are bitwise identical and match the rational oracle. Each contains133985 lossless Float64 words:28417 exact prefix words,69408 output pixels,28224 actual GPU input echoes and7936 guards. Unsigned values up to2^32−1 are exactly representable in this encoding; read-only replay reconstructs every original typed input byte. Payloads retain256-byte alignment through256-byte leading padding, with16 trailing typed guards. Exact output, input, guard, convex-range, full-range nearest and bilinear-identity checks are retained. The computed absolute error is zero; uncomputed relative error is null.

## Mapping and performance scope

All four versions retain source groups of eight channels per spatial iteration and the original29184 blocks×256 threads. These named small images require at most1344 active logical iterations, so most blocks do no pixel work. The source grid-strided loop remains, but repeated-stride work is not exercised here. No large-image bandwidth, roofline, perceptual quality or antialiasing claim follows.

KA and JACC use their own dispatch/global indices and ordinary Julia unsigned/Float32 arithmetic, with no CUDA-specific device intrinsics. Only CUDA backends have been verified. Neither version needs device shared memory, barriers or atomics; JACC explicitly requests zero shared bytes. KA's kernel object and JACC's LaunchSpec are cached outside timing. Julia hardcodes the original caller's selected true coordinate flags, whereas native receives runtime booleans; false-flag versions are not authored in this selected scope. Native's bilinear unroll-disable pragma and Julia's loop may produce different machine code; final instruction/resource behavior is unprofiled.

Each trial times24 paths of three launches plus completion, with per-path clocks retained. Reset, allocation, readback, full rational-oracle checks, serialization and all-path warmup are outside. Five checked trials remain and the mean uses2–5, with no outlier removal. This is a conditional, small-image, fixed-grid comparison and cannot rank abstractions independently of source mapping.

The first current CUDA.jl/KA/JACC candidate passes its first compiler/check invocation; there is no portable kernel repair. The original native numerical failures are separate N1 origins and the historical Julia tie difference is a retrospective source finding. Shared math, correction, witnesses, fixtures and native adapters are attributed once to the reference phase and reused KA then JACC. Prospective windows include repeated/cached context and coordination and overlap the outer workflow; do not sum them or infer exclusive billed or intrinsic cold-start costs. Historical creation cost is unknown.
