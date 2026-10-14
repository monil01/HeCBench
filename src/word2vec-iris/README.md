word2vec IRIS port
=================

The original vocabulary, sampling, host batching, options and defaults are
retained. Device embeddings, sentence batches, negative sampling table, RNG
state and sigmoid table are IRIS DMem buffers. CBOW runs on the HIP backend.
Because the IRIS task API does not expose the original dynamic shared-memory
launch size, each block uses a disjoint DMem scratch slice with the original
layout and block synchronization. This is a portability/performance compromise;
the CBOW arithmetic, indexing and block geometry remain the same.

The original training kernel has concurrent non-atomic embedding updates and
warp-synchronous reduction assumptions. Numerical verification must compare
actual vectors against native HIP, accounting for native variability; successful
execution alone establishes no numerical pass. RUN_ARGS selects a bounded
corpus workload; make run retains the original full text8 training args.

HECBENCH_WORD2VEC_BLOCKS optionally bounds the launch to the first requested
blocks, preserving the original default when unset. Numerical validation uses
one block to avoid the original non-atomic embedding races, one CPU training
thread and one iteration on the first megabyte of the original text8 corpus.
This covers original CBOW arithmetic, all 128 lanes, six-sentence batches,
200-dimensional vectors and 25 negative samples. The unrestricted multiblock
comparison failed, as did native-versus-native repeatability. No full training
equivalence or performance equivalence is claimed from this bounded check.
