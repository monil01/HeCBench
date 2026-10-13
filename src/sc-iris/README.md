IRIS dmem port of the default DP=no stream compaction. Original block scans,
CPU/GPU static partition, merge, data seed123, CLI and default sizes/repetitions
are preserved. DMem.update follows each host reset; input/output alias uses rw.
Fixed257-int shared storage accommodates original supported<=256 threads.
GPU submission completes before the original CPU worker is launched; this
serializes independent static partitions and is a performance compromise.
Optional CUDA DP=yes uses coherent managed-memory atomics and is not supported
by this dmem port; it is not the source default.

SC_DUMP optionally writes all compacted integers. An isolated native copy adds
only the same artifact hook. Bounded validation checks complete artifacts and
original Test Passed verdicts for -a0 and-a0.25, varied compaction factors,
warmup/repetitions, with defaults otherwise unchanged. CUDA backend unvalidated.

