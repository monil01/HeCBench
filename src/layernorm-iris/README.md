IRIS DMem layernorm port preserving batch/sequence/channel/repeat arguments,
seed0, three kernel algorithms, six block sizes32..1024, and all output/mean/
reciprocal-standard-deviation reference checks with1e-5 plus rounding tolerance.

HIP sibling reductions and cache helpers are retained. The two unroll variants
have exported C symbols for IRIS lookup. Data memory uses explicit input
updates, typed scalar dispatch, rw intermediates, output flushes, and explicit
runtime finalization. No native host memory or launch operations remain.

Default `8 1024 768 2000` is preserved; also pass the original second shape
`8 1024 12288 2000` via RUN_ARGS. Bounded validation exercised `2 3 64 2` and
`2 3 67 2`, all kernel/block/check combinations, against native HIP.
