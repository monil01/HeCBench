# Full softmax Jacobian study

## Scientific operation and accuracy

This machine learning operator materializes the complete matrix J = diag(y) − yyᵀ, where y = softmax(x). It is distinct from the O(D) vector product normally used during training. The named cases use batch size 8, dimension 128 or 512, and 256 threads per block. The original dimensions 2048/8192, other block sizes, batch size 1, and the nonmultiple-of-four fallback remain unmeasured. Inputs are synthetic logits in [−3, 3], generated exactly with integer arithmetic and division by 32. No scientific dataset or full model is evaluated.

Every trial checks all 131,072 or 2,097,152 Float32 entries against an independent Float64 formula and both GPU references, with absolute tolerance 1e−6. This is stronger than the original 1e−3 tolerance; observed maximum errors are below 6e−9. The original native Float32 CPU reference also passes. Eight centered finite differences check selected off-diagonal derivatives of a Float64 softmax, using step 1e−5 and tolerance 1e−9. Additional checks require finite values, positive diagonal entries, negative off-diagonal entries, symmetry, zero row sums, and a nonnegative quadratic form for one probe vector per sample. That probe does not constitute an exhaustive eigenvalue test. No PyTorch or automatic differentiation run is inferred from source comments.

## Mapping and reference correction

The historical Julia k1 kernel is genuine and its text is preserved byte for byte. Its k2 label called the same k1 kernel. The current reference restores a distinct k2 path with warp reductions and groups of four output components. Original CUDA device kernels remain unchanged.

Native k2 explicitly loads and stores Vec4 values. Julia expresses four scalar components; actual generated vectorization is unknown. Julia also adds a barrier after reading the maximum scalar, before reusing warp partial storage. Source inspection identified a potential lifetime hazard in native k2, but all 20 native study trials passed. This is a disclosed synchronization adaptation, not an observed native failure or a compiler feedback repair.

K1 uses shared-memory trees through framework kernel APIs. K2 uses explicit CUDA warp shuffle calls in all Julia versions and is classified as hybrid_device_intrinsics. Per-configuration ontology records distinguish these paths. Cross-backend warp portability was not demonstrated; no unsupported API failure is invented. JACC additionally stages a zero template with 384 or 768 elements and an initial barrier. All four versions reserve the original host-selected 1,536 or 3,072 shared-memory bytes, although k2 uses only eight warp partial slots. Registers, spills, occupancy, and generated memory instructions were not measured.

## Timing and effort

All four versions use explicit warmup, five in-process synchronized trials, the mean of the last four, and ten overwrite operations per trial. Allocations, resets, transfers, CPU formulas, finite differences, invariants, and artifacts are outside the timed region. Eight blocks form a small grid. Output footprints of 0.5 MiB and 8 MiB describe a capacity contrast; they do not prove a cache or DRAM bottleneck without counters. Vector memory access, barriers, and template staging also limit causal attribution to the framework alone.

KA and JACC each passed one candidate with four configuration checks, two generic device kernels, and zero repairs. Shared reference work comprises one source revision and four native builds/reference checks. It restores k2 while preserving k1. Historical generation effort is unknown. The KA usage window closed before the first JACC draft; JACC sees successful KA feedback. Source character proxies and context/cache-inclusive consumed usage remain separate. The broad workflow window overlaps child phases and must not be added to them or interpreted as monetary billing.
