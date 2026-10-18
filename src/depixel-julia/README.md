# depixel: conditional color-connectivity graph pipeline

The study implements both source stages: Float32 RGB-to-quantized-YUV8-neighbor connectivity, then local crossing decisions from color frequency, islands and capped curve scores. Seven images cover rectangles, checkerboards, a curve transition witness, an isolated pair, colored palettes and chains. Equal-weight crossings retain both diagonals, as in the source policy. This is not a complete vectorizer, guaranteed planar graph or image-quality evaluation.

## Source defects and conditional reference

Before any GPU run, original-body CPU-ASan reproduces a rectangular out-of-allocation read on valid source-produced checkerboard connectivity. A valid thin-path image reaches a missing-break transition that moves to a non-neighbor. A packed-field witness shows that wholeword island comparisons include color bits instead of only connectivity. Original sources and Makefile are archived intact; reference_correction.patch changes two extent bounds, one break and two low-byte tests. No other heuristic is redesigned. The corrected C++ CPU pipeline matches an independent adjacency-list graph oracle under ASan on all seven cases; an original-safe uniform-image CPU control also matches. No known-invalid original GPU run is performed.

The native GPU study explicitly includes the corrected header and carries native_reference_repaired plus canonical provenance before first measurement. Its timings are excluded from untouched-native aggregates. Historical Julia was a constant smoke placeholder; its original bytes/Makefile are preserved, while the new study is a genuine two-kernel implementation.

## Exact data and numerical contract

Every trial retains27198 UInt32 words:9998 input/parameter words plus all3328 intermediate graph pixels,3328 final pixels and actual GPU RGB echoes, including560 physical guard words. All twenty artifacts agree exactly with both references and the independent graph oracle. The oracle uses explicit undirected neighbor lists, geometric degree2 traversal with reciprocity checks, degree1 and local color-frequency scores, and the source cap/tie policy; it does not import GPU helper code.

All selected Float32 palette colors have scaled YUV quantization-bin margins at least.0224036, exceeding a conservative32eps(Float32)*255 allowance of.000972748 for finite normalized operations. FMA/association differences cannot alter bins in these selected fixtures; integer output tolerance remains zero. Arbitrary RGB threshold boundaries, nonfinite inputs and size tails are unverified. Actual input layouts are the same12-byte-per-pixel AoS: native float3 and Julia flat Float32 triples.

## Timing and effort

Each of five outer trials sums seven host-inclusive100-repeat two-kernel batch regions divided100. Fixed inputs repeat within a batch; allocation, H2D, output reset, compilation warmup, readback, validation and serialization are excluded. Only trial1 timing is omitted from the mean; every per-image duration is retained. This is a named component protocol, not the source's changing-image CLI workload.

CUDA.jl/KA/JACC use genuine framework launches and one shared ordinary Julia math helper, without CUDA-specific device intrinsics. JACC explicitly requests zero shared bytes. Source duplicated walkers are factored into one inline helper; compiled control flow/resources remain unprofiled. No bandwidth/roofline/branch or intrinsic framework-overhead attribution follows from these composite host timings.

The first reference GPU candidate compiled, but a shared Printf argument error prevented structured trial capture. That first numerical outcome is unknown and its logs/source are preserved. Removing the extra formatting argument was a T1 measurement-adapter repair; GPU math was unchanged. The second reference program check passes all five; KA/JACC first candidate/checks pass. Historical cost is unknown; shared oracle/math preparation is attributed once and prospective phase usage is ordered/marginal, not cold framework ranking.

Use audit_saved_outputs.py for CPU-only full saved-array/hash/timing rechecks and the central serialized GPU6 runner for future GPU work. Root alone commits.
