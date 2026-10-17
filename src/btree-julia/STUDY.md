# btree: concurrent ordered GPU map

The historical Julia source is a smoke marker that prints a zero-key PASS. The repaired reference and two targets implement the actual native warp-cooperative B-link tree, including atomic allocation, node locks/fences, root and nonroot splits, side-link traversal and warp work queues. The original native template kernels are unchanged. A smaller bounded allocator is passed through their existing generic allocator interface with the same address arithmetic and atomic allocation in all four models.

## Complete semantic verification

Cases insert 256 and 4096 unique shuffled keys with independently keyed nonzero values. Every key in the range zero to twice the tree size is queried exactly once, so both hits and misses are exhaustive. Every occupied user leaf key/value pair is also copied, sorted and checked against an independent integer dictionary. All five trials of all four models match 17,416 exact serialized input words and 17,408 output words. Canonicalization intentionally removes tree shape, slot order and allocator IDs; this is complete semantic dictionary evidence, not raw physical-tree identity or a proof for all concurrent schedules. Native reserved key1 sentinel is excluded from user records.

## Scope and mappings

The fixture uses safe31-bit keys/values and warp-aligned query counts. Native search writes results unconditionally for an active final warp; the source-scope record shows why partial query warps need a separate repair or padded allocation. No runtime failure is claimed for that unmeasured regime. Delete/range APIs and conflicting duplicate insertion semantics remain outside this benchmark's exercised insert/search path. Allocation, copies, pool reset, root initialization and output canonicalization are outside the matched named build+search timer. This kernel-region study does not reproduce native main's end-to-end allocation/copy timing.

CUDA.jl uses its own exported CUDA warp, atomic and fence APIs and is a framework device kernel. KA and JACC call those backend APIs and are hybrid device mappings. No cross-vendor capability is inferred. Both use256-thread blocks; JACC explicitly requests zero dynamic shared memory. No counters establish a performance mechanism.

## Compiler feedback and effort

CUDA.jl's first reference candidate lost internal-node flags because native signed ballot arithmetic right shift was translated as unsigned logical shift. An illegal node pointer prompted a diagnostic bounds candidate, then a signed reinterpretation repair. The minimal witness is `0x80000000 >> 2`: native signed bits become `0xe0000000`, versus the wrong logical `0x20000000`. Three candidate/check processes include this one diagnostic variant and one actual semantic repair. KA first failed a host backend namespace lookup before any device JIT; its second program candidate compiled and passed. Shared instrumentation also required two canonicalizer repairs, and auxiliary CPU metadata/path errors are retained separately from target compiler failures.

Five trials are retained, with trial1 ignored and the last four averaged. Sum of two build+search regions in microseconds: {"cuda": 675.58425, "cuda_jl": 649.46825, "ka": 637.85825, "jacc": 736.91625}. Result: `/home/imo/HeCBench_julia/.porting-state/portable_julia/runs/20261017T222307658331Z_btree/result.json`. Prospective activity windows include cached/repeated context, same-benchmark witness/annotation work, feedback, coordination and waiting; they are not exclusive coding or billed costs. Partial final-source proxies exclude common algorithm code.


Prospective rawusage intervals include cached/repeatedcontext and coordination; the parentworkflow contains its childphases and must not be summed with them. Exclusive billed translation cost remains unknown; source-size proxies remain separately labeled.
