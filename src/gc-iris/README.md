IRIS DMem graph coloring port preserving arguments, default internet.egr /
100 repeats, graph invariant verification and color summaries. The HIP kernels
retain the sibling's 64-lane ballots/shuffles while colors remain 32-bit words.
Only HIP backend source is provided for this wave-size adaptation.

All mutable intermediate buffers use iris_rw, and the worklist-size reset
explicitly updates DMem each iteration. Host read-only HIP device-property
queries retain the baseline occupancy-derived block count. Allocations, copies
and kernel submissions use IRIS data memory and tasks. Final colors flush
before verification; finalization precedes static teardown.

Validated the unchanged internet graph with two repeats against native HIP,
including graph invariants and exact color histogram. Defaults are preserved.
