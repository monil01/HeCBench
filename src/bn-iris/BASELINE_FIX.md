The lookup named LG is actually log-factorial: LG[n]=sum(log(i),i=1..n).
Its missing zero entry is therefore log(0!)=lgamma(1)=0, not lgamma(0).
Empty contingency-table cells read this entry. Initializing LG[0]=0 identically
in CUDA/HIP/IRIS defines the intended score without changing the recurrence.
Untouched originals and earlier failed attempts remain in local porting state.

The argmax reduction now stores integer winner indices separately from float
scores, with deterministic first-rank tie breaking. Previously tied negative
scores could be converted to invalid thread indices, and the O3 native graph
search timed out while O1 completed all100 iterations. The new reduction uses
an additional256 shared integers; this changes reduction performance, not the
score enumeration or mathematical argmax. Native and IRIS use the same repair.

Parent decoding stops on both0 and-1 sentinels (inserting the leading index0
leaves a trailing0 for a three-parent set). Exclusion-index decoding uses <=node
to map that node's one-based predecessor back to the actual prior node, avoiding
self-edges. The CPU reference uses the same corrected mapping. All100 search
iterations, RNG seed2 and production45x600 defaults are retained.

The CPU reference's five-element parent array is initialized to negative
sentinels. H_findComb fills only four slots; insertion of a leading0 otherwise
copies an undefined fifth slot for three-parent selections, adding a spurious
fourth graph edge. Independent graph scores caught this despite native/IRIS
artifact equality. No printed likelihood or tolerance is adjusted to conceal it.

ROCm7.0.1 clang20 qualification: native O3 repeats a graph-kernel stall after47
node selections, including after the defined reduction repair; O1 completes all
100 search iterations. IRIS HIPFLAGS defaults to-O1 and native validation uses
OPTIMIZE=no EXTRA_CFLAGS='-DHECBENCH_BN_VALIDATION -O1'. This is a toolchain
portability/performance qualification, not an algorithm, workload or tolerance
change. Original CUDA/HIP Makefile defaults remain unchanged.

verify_scores.py independently enumerates parent combinations, counts each binary
contingency table, and uses double-precision math.lgamma to check every bounded
local score. This is validation-only CPU work, not an execution fallback.
