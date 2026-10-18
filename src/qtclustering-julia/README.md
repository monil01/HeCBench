# Quality-threshold clustering: complete compact-storage pipeline

The historical Julia version marked an array and printed fabricated iteration counts, cluster sizes and timing components. Its source and Makefile are archived. The current CUDA.jl, KA and JACC studies execute the actual compact-neighbor clustering algorithm on the GPU. All five native kernel bodies remain unchanged through their original header includes: degree computation, candidate generation, cardinality reduction, winning-cluster recomputation/active-list compaction, and clustered-mask update.

## Scientific contract

For every remaining seed, the source grows a candidate by adding the point minimizing the maximum distance to the candidate, with lowest point ID on a distance tie. It selects the largest candidate, retaining the earliest active seed on a cardinality tie. This is the source greedy complete-link heuristic, not a globally optimal clique or partition algorithm.

Three exact dyadic one-dimensional clouds embedded in the original two-dimensional Euclidean domain contain192 points in67/65/60 cliques,128 points spaced by0.25, and65 points containing a duplicate pair plus63 isolated points. Compact sorted nonself neighbors use strict distance<1, with−1 and FLT_MAX padding. The independent oracle constructs full pairwise coordinate geometry and implements the mathematical greedy candidate rule, without reproducing the compact cache or lane reduction. All chosen members, order, seeds, cardinalities, active remaining lists, degrees, masks and cluster diameters are checked at every iteration.

The source produces3,32 and2 iterations. In the isolated-point control, it emits the pair and one singleton, then stops at cardinality<=1 with62 isolated points still in the actual remaining list. No fabricated explicit complete partition is substituted for that source behavior. Full source random4096-point CLI, distributed MPI, degree>768 fallback/prefetch-tail behavior, and global optimality are outside certification.

## Mapping and observability

All versions retain64-thread cooperative source workgroups, a one-thread global cardinality reduction and source compact row-major arrays. Float32 distances/cache/threshold, Int32 graph/cardinality/result buffers and Int8 masks are retained. All fixture distances are exactly representable dyadic values, so the declared acceptance is exact: atol=rtol=0.

Each Julia model genuinely dispatches its own framework kernel, using explicit CUDA backend extensions for static shared arrays, block barriers and source cooperative block indexing. These are CUDA-specific hybrids. JACC explicitly requests zero dynamic shared memory; static backend shared declarations are separate. Julia stores the twelve prefetched candidates in an isbits tuple, versus twelve named native variables; compiled register/local-memory allocation is not profiled.

Julia compaction uses a uniform masked tail and an additional shared-read lifetime barrier before in-place writes. This is a disclosed safety/synchronization mapping covariate. The original native code remains unchanged and every captured native result passes the independent oracle; a static possible interleaving hazard is not reported as an observed native numerical failure. Otherwise-unread output/workspace locations are initialized to deterministic canaries for observation outside timing, without changing native algorithm inputs.

Each full artifact has798063 Float64 lossless words:27790 exact input-prefix words, complete per-iteration logical states, all mutable workspace snapshots including stale tails and caches, actual GPU input echoes and432 guard fields across three cases. The independent oracle covers scientific states; cache/stale-tail contents are separately compared with both GPU references. All raw bytes are retained. A read-only replay checks every saved artifact, actual Float32 input bits, masks, degrees, members, remaining lists and guards. No NaN is accepted. Program error metrics are null because the program computes Boolean exact acceptance, while the verifier independently records actual full-array maximum errors.

## Timing and effort

The named region sums all151 degree/candidate/reduction/trim/update kernel launch+completion intervals over the37 convergence iterations. Degree and global reduction are included equally for all models, unlike the original printed qtc+trim+update sum. Per-stage times and labels remain in every trial. Card/result host readback, reset/allocation, fixtures, oracle, serialization and complete-pipeline compile warmup are outside these intervals. This is not full CLI end-to-end time or pure device-only profiling. Five checked trials remain; only trial1 is excluded from the mean, with no outlier removal.

Each model's first current GPU candidate passes its first check. Historical marker/printed-statistics failure is a CUDA.jl reference S1 finding, not a portable compiler failure. Shared algorithm/native adapter/oracle work is attributed once to the reference phase and reused in KA-then-JACC order. Prospective usage windows include repeated/cached context and coordination and overlap the outer workflow; do not sum them or infer exclusive billed costs or intrinsic cold-start framework rankings. Historical creation cost remains unknown.
