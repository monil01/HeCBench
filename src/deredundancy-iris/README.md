# deredundancy, IRIS DMem / HIP

The 14 original device kernel bodies and CPU option parser, FASTA reader,
length sort, index sort and FASTA output are retained. C-linkage exports replace
native launches, and all device storage uses DMem. Original options are pairs
`i inputFile`, `o outputFile`, `t threshold`; defaults remain testData.fasta,
result.fasta, threshold0.95. The native Makefile default input path is preserved.
Thresholds select word lengths4--7 by the original arithmetic; no algorithm or
threshold bug is repaired. Input data is not committed.

`make`; `IRIS_ARCHS=HIP make run RUN_ARGS='i /path/testData.fasta'`.
Index storage is flushed for the original CPU sort, then `DMem.update` publishes
the sorted host values. Partial writes and cluster/table updates are read/write.
Representative selection uses the original single-thread device algorithm.
Source-unused padding in host output backing vectors is zero-initialized;
all defined sequence/packed/index/cluster values retain source computation.

`DEREDUNDANCY_DUMP=/path/file` adds a lossless observer of every final cluster,
all lengths/offsets/words/gaps/base counts/cutoffs, the full table, every actual
normalized base, defined packed word, index and merged order. It deliberately
excludes source-uninitialized padding, which is not algorithm output. Native
HIP validation uses an isolated copy with the same observers only. Complete
result.fasta bytes and all defined binary values are compared exactly.

`DEREDUNDANCY_GPU_LEASE=1` serializes the GPU preprocessing and clustering
phases, with wait<=300 seconds and each held lease<=90 seconds. Clustering yields
at completed representative boundaries every16 iterations or after20 seconds,
with a bounded1800-second validation phase. This adds synchronization/observation
only and leaves the algorithm, input, thresholds and stopping rule unchanged.
FASTA preparation,
CPU sorting, artifact writing and external verification are outside the lease.
Validation uses the recovered original DVC FASTA and bounded original records
for all four threshold branches; native originals are untouched.
