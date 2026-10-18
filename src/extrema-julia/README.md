# extrema: actual typed one- and two-dimensional relative maxima

Historical Julia source/Makefile are archived intact. They launched only a smoke marker and printed96 fabricated timing lines without computing extrema masks. New CUDA.jl/KA/JACC versions compute the actual native kernels. Native kernel bodies are unchanged. Each current reference/target candidate passed its first compilation/check, with no current repair. The historical S1 finding is source evidence; original creation attempts/tokens are unknown.

## Boundary, ties and dtype

Source predicate requires data>every positive-offset neighbor and data>=every negative-offset neighbor. The asymmetric rule picks the positive end of a plateau: [9,9,1] gives [false,true,false] for order1/clipping. It is not a strict-both-sides maximum or a peak-prominence/noise-filtering algorithm. Clipping clamps coordinates; wrap mode adjusts one period only, so all selected wrap orders are<=axis length. No out-of-domain wrap order is certified or executed. A length1 axis always compares strictly with itself and yields false.

Actual Int32, Int64, Float32 and Float64 inputs include plateaus, boundary peaks, Int64 extremes, floatNaN/Inf/-Inf and signedzero controls. Ordered comparisons with NaN are false; NaN is an input control, never an output accepted through a NaN tolerance rule. UInt64 artifact words retain raw typed bits, with32-bit representations zeroextended. This avoids rounding Int64 extremes through Float64 or canonicalizing NaN/signzero inputs. Native sizeof(bool)==1 is checked; Julia stores UInt8(0/1), the identical physical mask payload.

## Verification scope

Named shapes1D257,2D17x31 anddegenerate1x19 cover all96 original type/order/axis combinations(orders1..128,clipping) plus36 wrap and16 degenerate-axis paths,148 total. They are not the original1M-element CLI runtime/input claim. An independent coordinate oracle uses clamping or mathematical modulo and explicit neighbor comparisons.

Each of20 captures retains123972 UInt64 words:3260 actual input/dimension prefix words and120712 full mask/inputecho/guard words. All57988 mask bits pertrial,4736 physical guards and actual typed input bits agree exactly with both references and the oracle. Per-path true counts also match. No masking, scalar checksum substitute or canonicalized output is used. Uncomputed floating error fields are null; actual artifact word mismatch count is computed(0). The runner computes exact dual-reference differences separately(0).

## Timing, traits and effort

All models match256thread1D gridstride and16x16thread2D ceil grids, with partial workgroups and explicit bounds. Five checked trials each sum148 host-inclusive100-launch batch regions divided100; only trial1 timing is excluded. Input construction/copies/reset, compilation, readback/oracle/serialization are outside. All raw samples, including spikes, remain. Every per-type/order/axis timing is retained instead of treating the mixed aggregate as one scientific workload. JACC explicitly requests0 shared bytes and includes cached LaunchSpec lookup in host timing. No profiling supports cache/load-reuse/occupancy/bandwidth or roofline attribution.

The shared comparison helper was authored during reference restoration and reused by KA/JACC through genuine framework indexing/dispatch. Device code uses ordinary typed Julia comparisons/indexing, with no CUDA-specific intrinsic. Prospective windows include repeated/cached context, annotations and coordination, overlap with the outer workflow and are not additive, exclusive billed usage or intrinsic cold framework costs. Historical creation cost stays unknown. Root alone commits; audit_saved_outputs.py checks all saved evidence without GPU trials.
