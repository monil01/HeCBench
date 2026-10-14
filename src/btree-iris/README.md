The original B-tree device algorithm is retained in map_warps.hpp.
HIP uses one 32-thread logical warp per block to avoid lock divergence between
two logical CUDA warps sharing an AMD wavefront. Shuffles use width 32 and
ballots select the relevant 32 bits. The original 4 GiB pool and input defaults
are retained; partial query blocks guard result stores.
There is no native HIP sibling. Validation uses the original CUDA key/value
and not-found reference checks with random_device initialization preserved.
