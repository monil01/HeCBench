# CCSD triples energies with IRIS DMem

Retains original coupled-cluster kernel expressions,2D16x16geometry,device-only
CUB/hipCUBBlockReduce and atomic double reductions. Original host aggregation,
CLI/default160400100,nkpass/memorycap,seededdrand48 and CPU reference remain.
IRIS initializes after all source input generation. Input DMems are read-only,
zeroed per-call scalar counters use read/write, and every output is flushed.
Optional HECBENCH_CCSD_TRACE records all four GPU energy contributions percall.
Native isolated baseline follows native Makefile's original CUDAhost sources,
with only identical tracing added. Validates bounded and defaultdimensions,
complete per-call energies,source1e-6CPUverdict and printed accumulated energies.
