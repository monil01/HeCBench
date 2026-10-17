# Keogh study

## Scientific meaning and fixture

The kernel computes a squared envelope-distance score for each normalized subject window. With a compatible query envelope and warping constraint, this is the squared LB_Keogh bound on squared dynamic time warping cost. The role of protective envelopes in DTW lower bounding is described by the [author's page](https://www.cs.ucr.edu/~eamonn/LB_Keogh.htm) and [primary paper](https://www.cs.ucr.edu/~eamonn/vldb2002.pdf). No full nearest-neighbor search, pruning-rate study, or physical application dataset is evaluated.

The named fixture uses query length 256, subject length 262144, and envelope radius 8. It supplies actual population means/standard deviations for each window and an ordered envelope around a normalized query. The reproducible binary generator and SHA256 provenance are retained. Standard deviations lie between 1.1352 and 1.1727. The original 20-million-element case and its independent random means, standard deviations and bounds remain unmeasured. Those arrays define kernel arithmetic, but they do not by themselves establish the scientific envelope/normalization assumptions. No native numerical failure is claimed from that distinction. Preprocessing is outside timing in every version.

## Numerical checks and mapping

Every trial checks all 261889 Float32 scores against an independent Float64 distance-to-envelope formula and both GPU references at the original absolute tolerance 1e−2. The observed maximum Float64 error is approximately 2.03e−5. The original native Float32 CPU reference also passes. Scores must be finite and nonnegative. An independent Python dynamic program checks radius-8 DTW costs for eight selected windows against every saved trial of all four versions. These eight domain witnesses supplement full score checks; they do not constitute 261889 separate DTW runs.

Native device code is unchanged. Historical Julia only touched one marker and printed zero sizes, zero time and PASS; that source is archived and the current reference correction is separately costed. All versions use 1024 blocks of 256 threads, a 512-float shared halo, and one serial 256-term score per valid lane. The output has 261889 entries: the last block has only one valid output. All lanes load and reach the barrier before the output guard. KA and JACC launch padded blocks so no lane exits before that barrier. Unused padding is not counted as numerical evidence.

JACC explicitly reserves 2048 shared bytes and stages an additional 512-value zero template with an initial barrier. Native, CUDA.jl and KA directly load the halo. These are ordinary framework device kernels, without a whole-operator vendor library or explicit backend-specific warp operations. Actual division, reciprocal hoisting, FMA instructions, registers, spills and occupancy remain unmeasured.

## Performance and effort

All four versions use explicit warmup, five synchronized in-process trials, the mean of the last four, and ten overwrite operations per trial. Allocations, resets, transfers, statistics/envelope construction, CPU formulas, DTW witnesses, and artifacts are outside timing. The roughly 4 MiB combined working footprint is much smaller than the original case. No scaling, DRAM bandwidth, compute bottleneck or framework ceiling follows from this case alone.

KA and JACC each passed one candidate, one check and one device kernel with zero repairs. Shared reference work comprises one semantic revision from the stub and one native build/reference check, plus fixture and independent oracle work. Historical generation cost is unknown. The KA window closed before the JACC draft; JACC sees successful KA feedback. Saved-source proxies remain separate from context/cache-inclusive consumed usage. Broad workflow and child windows overlap and must not be added together or interpreted as monetary billing.
