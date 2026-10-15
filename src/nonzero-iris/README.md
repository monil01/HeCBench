# IRIS nonzero

Uses HIP-only IRIS DMem tasks for reduction, stable selection, and coordinate
conversion for all five original data types and both warmup/measurement passes.
The vendor DeviceSelect operation is replaced by a serial GPU offset pass and
one GPU thread per 256-element chunk for stable compaction. This preserves the
complete lexicographic output but is deliberately not performance-equivalent to
hipCUB. A whole-output CPU traversal checks every coordinate and its order.
Original CLI, seed, input distribution, and Makefile default are retained.
