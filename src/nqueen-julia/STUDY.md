# N-queens: full original15/depth7 prefix-partitioned DFS

This is combinatorial constraint counting over an implicit irregular search tree, not a physical simulation or a game application. It places one queen per row, excluding repeated columns and diagonal conflicts. The native case N=15, initial depth7 is retained in full:1,897,702 legal CPU prefixes, one suffix DFS task per GPU lane. The historical Julia file included unrelated mcpr code; it is preserved and replaced with the actual iterative search.

## Independent complete evidence

The original source prefix generator is compared against an independently implemented recursive bitmask enumeration for exact prefix order, controls and first seven signed-byte columns. Five unused root-board fields are initialized to zero for canonical input bytes; the GPU source reads only the first seven prefix fields. An immutable Julia root type matches the native16-byte UInt32+12Int8 ABI. The named harness reserves only actual prefix capacity outside timing instead of the source's75,580,635-entry host buffers. Neither change repairs native device arithmetic.

An independent bitmask suffix solver computes every prefix's accepted-node and solution counts. All3,795,404 UInt64 output values are checked exactly in every trial, plus all30,363,232 actual root-input bytes serialized as UInt64 words. A separate CPU audit verifies all20 full artifacts byte-exactly, every prefix's column/diagonal legality, masks, unused fields and complete counts. The source constants are supplemental:2,279,184 solutions and171,129,071 accepted nodes, including2,466,109 CPU prefix nodes. No full solution-board output arrays are invented; this source emits per-prefix counters.

## Mapping, work distribution and timing

Native private char[20] is represented by a standard StaticArrays MVector{20,Int8}, with the same iterative backtracking and non-short-circuit vertical/diagonal scans. UInt32 masks and true UInt64 counters retain integer semantics. Native64-bit shift literals narrow into the UInt32 mask; Julia32-bit shifts are algebraically identical for selected columns0..14. Native unroll hints and actual private-array/register/stack lowering may differ and are unprofiled. Every framework launches128 lanes per block, one source-ordered prefix per lane. No block synchronization, atomics or vendor operator calls occur; KA/JACC use own framework indices and pure Julia/StaticArrays helpers, with JACC explicitly requesting zero shared memory. Only the NVIDIA backend is tested.

The verified GPU suffix tree contains168,662,962 accepted nodes. Per-prefix accepted-node counts range0–635, mean88.877, median77, p99=275 and CV0.656. Solution counts range0–33. These are useful observed irregularity traits, not total legality checks, instructions or measured warp divergence. A zero accepted-node task still tries rejected placements. No ordinary floating-point roofline position or causal bottleneck is inferred from these integer/control-flow counters.

One explicit cold native feasibility probe ran the full case, matched every prefix count and took about110.9ms; it is retained separately and may include module loading. Before primary timing, one complete search per trial was selected, replacing source timer repetition100 without reducing the search. Each process warms one full search before five in-process trials; only launch plus blocking device-wide completion is timed, and the last four trials are averaged. Prefix/oracle generation, copies/resets, validation and serialization are outside timing.

| CUDA (ms) | CUDA.jl (ms) | KA (ms) | JACC (ms) |
|---:|---:|---:|---:|
| 101.709 | 136.378 | 136.506 | 136.295 |

All samples and dispersion are retained. Similar Julia-family timing does not establish a universal abstraction ranking; private-state/control lowering requires profiling. Other board sizes/depths, source fixed storage limits and whole CLI host allocation/generation time are outside the verified contract.

## Effort and audit boundaries

Actual reference restoration passed one candidate and one device-JIT check with no compiler repair. Its shared DFS/legality helper is separately costed from the small portable wrappers. KA and JACC each passed their first kernel candidate and one actual device-JIT check, with zero repairs. KA also encountered a verifier preflight metadata mismatch before Julia launched: correcting archived contract prose violated exact reuse-string matching. The archived phrase is retained with a hash-bound prose erratum; restoring the immutable contract plus explicit erratum fixed metadata without changing code, acceptance or timing. This tooling rejection is not a kernel compiler attempt. A separate prose typo audit likewise changes no execution.

The KA phase window includes descriptive CPU subtree-statistics work and preflight handling, so it is not exclusive kernel-code consumption. Historical cost is unknown; shared reference/helper/native diagnosis/oracle, target drafts and annotation are distinct phases. Context/cache/reasoning/coordination are included, broad/child windows overlap, and exclusive billed translation cost is unknown.
