# IRIS logic-rewrite

Complete original graph transformations retained in native/ (source provenance,
not an execution fallback). All persistent GPU buffers are IRIS DMem, with
addresses resolved by DMem task binding. Custom kernels launch through the IRIS
vendor-kernel API from IRIS tasks. Thrust GPU operations use IRIS-provided HIP
streams through tools/iris_vendor_bridge.cpp. Hidden DMem bindings track every
live buffer's input/output ownership for each task. Callbacks synchronize and fail
loudly on errors. No shared runtime edits or CPU graph-computation fallback.

P1: conservative read/write dependencies on all live DMem buffers, synchronous
tasks, and host staging for partial buffer copies and memsets change performance
but not graph math. IRIS owns context lifetime instead of native final reset.
Original CLI, default graph and complete resyn2 sequence remain unchanged.
Port-local graph_hash renaming avoids IRIS header namespace pollution.

HIP heap/stack limits, resource queries and error-state calls also execute
inside IRIS worker tasks: configuring the main thread's native HIP context
does not configure the context that actually runs the DMem kernels.

Build with make; run with IRIS_ARCHS=HIP make run. Backend wrapper and liblogic.so
must be in the run directory. Timing alone is not correctness evidence; compare
full graph functions against original input and native HIP.

Native baseline repair: the truth-table traversal partitions temporary storage,
heap allocation and lane indices into groups of 32. Make WarpScan's logical
width explicitly 32 in CUDA, HIP and IRIS. The HIP default is 64 on gfx908;
using it with 32-wide allocation groups corrupts prefix offsets, particularly
for partial waves. Untouched source snapshots and failing native small-graph
runs are retained in local evidence. CUDA's original width remains unchanged.

Validation qualification: original K12 refactoring on the derived bounded
subcircuit exceeds native fixed scratch/subgraph capacities. Those resource
limits are retained, not silently enlarged. Validate the complete resyn2 path
on this subcircuit with its original `-K 6` CLI option. The default K12 resyn2
on the original full graph remains unchanged; default-size IRIS testing hit a
180-second limit and is not a completed numerical/performance proof.
