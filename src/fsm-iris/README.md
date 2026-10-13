IRIS DMem FSM port. Defaults, seeds, compile constants, genetic algorithm and
host FSM verification are preserved from CUDA. GPU reset replaces cudaMemset
on every repeat. Tasks contain reset, FSM and Max kernels; best is flushed.
All device launches require an enumerated HIP GPU. Default trace length 65536;
RUN_ARGS accepts a bounded length for validation. Native PASS and hit counts
are numerical evidence. Kernels have a 32768-byte state per thread and can
be expensive even for short traces; execution timeouts are blockers, not PASS.
