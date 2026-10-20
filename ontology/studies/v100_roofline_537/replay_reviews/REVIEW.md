# Kernel replay safety review

CPU/source-only. Safe is a bounded static per-kernel host-dependency/state-restorability assessment, not empirical profiler correctness, whole-ROI coverage, or unconditional permission. Unknown and unsafe must not be autoexecuted.

## Current collector

Snapshot `d30c541274ac084b0f291b16277d1a5f9215a5685d9916bf0157d26296493602` hardcodes `--replay-mode kernel`, cache-control all, and first invocation per mangled function/geometry. No per-case replay override is implemented in this reviewed snapshot. Landlock prevents external writes; it does not make replay computationally faithful.

## Per-case verdicts

| Case | Models | Static verdict | Scope |
|---|---:|---|---|
| atomicSystemWide |4|unsafe|kernel replay cannot preserve same-allocation CPU/GPU coexecution contract|
| ans |4|safe|static host-participation criterion only; full device memory checkpoint/restore and scientific audit required|
| bsw |4|safe|static per-kernel state replay, not whole host-feedback ROI coverage|
| base64e |4|safe|individual independent encoding kernels only; serialized profile is not concurrent pipeline performance|
| pad |4|safe|selected static partition has separate host/device storage; intra-GPU spin progress remains an instrumentation caveat|
| sc |4|safe|selected static CPU/GPU compaction partitions own separate buffers; kernel-only counters|
| sss |4|unknown|two-stream retained scientific host controller; no automatic replay approval|

### atomicSystemWide

GPU system atomics and CPU atomics modify the SAME managed/UnifiedMemory allocation while the device launch is outstanding. Device-only replay does not repeat host participation for every metric pass. Restoring GPU-visible state is insufficient to reconstruct CPU/device interleavings; even successful process exit is not unchanged coexecution.

**Required:** Do not use current multi-pass kernel replay for this contract. Separate reviewed application replay must reproduce fresh CPU execution, managed allocation and property checks each pass; kernel matching/determinism and output isolation must be reviewed. Single-pass collection is a separately declared diagnostic, not automatic approval.

**Alternative:** Application replay with fresh per-pass output roots; or instrument one real execution with a separately approved single-pass metric set. Keep property rather than raw atomic outcome identity.

### ans

All six work arrays are cudaMalloc device buffers. CPU synchronisation-flag readback and prefix sum occur BETWEEN kernel launches using blocking cudaMemcpy; no selected kernel waits for a concurrently executing host response. Phase2 convergence changes successive kernel input states, so its first invocation is only one convergence state.

**Required:** Require complete restoration of device-written sync_points/sizes/flags/output for each metric pass, unchanged callback stream ordering, successful final exact decode and scratch/guard CPU audits, and no profiler warnings. Source review does not prove inter-block convergence under every instrumented schedule. Inventory every phase and convergence-state role before ROI aggregation.

**Alternative:** Application replay if checkpoint/restore or instrumented convergence fails, preserving original loop and full audits; NVTX stages/explicit reviewed invocation selectors for phase coverage. Never replace adaptive host convergence by fixed iteration count.

### bsw

Forward output endpoints are copied to host and synchronized before computing reverse launch geometry. CPU feedback is between launches, not a live response required by a running GPU kernel. Device buffers use cudaMalloc/CuArray rather than shared CPU atomic state.

**Required:** Restore all mutated device arrays before extra passes. Preserve forward result used to choose reverse threads, exact outputs/coordinates, and full independent/guard checks. Bind both direction/geometry/state roles: first-per-name-and-geometry can omit reverse if geometry collides, and cannot certify host packing/transfers.

**Alternative:** Reviewed direction-specific invocation/NVTX selection or application replay if restoration/output checks fail. Keep source host-feedback ROI time from original unprofiled measurements.

### base64e

Four streams operate on disjoint 768-byte input/1024-byte output slices; pinned host input remains stable until each stream completes. No CPU worker must respond during kernel execution. Device input/output state is replayable, but serialization and cold caches remove the intended overlap.

**Required:** Keep each stream offset/padding case, all stream completions and full Base64 output checks. Require device-memory restore and no warnings. Do not infer concurrent four-stream throughput, transfer overlap, or whole ROI roofline from serialized first-per-geometry kernel counters.

**Alternative:** For overlap-sensitive questions use reviewed application/range replay supported by installed NCU, or separate CUDA timeline instrumentation. Application replay alone can still serialize selected launches; verify tool behavior rather than assume concurrency preservation.

### pad

Selected GPU flags/data are cudaMalloc/CuArray copies, distinct from CPU std::atomic/vector allocations. CPU threads do not service the GPU flag array during the selected kernel. GPU inter-block flag chaining and in-place padding require state restore; generic system-atomic source variants do not imply the selected copy shares memory.

**Required:** Restore entire in-place device data/flags, preserve static alpha/cut and kernel local state. Full flag/data/guard checks mandatory. Instrumentation can alter progress of spin-wait kernels, so timeout/failure remains inconclusive rather than proof of framework limitation. CPU concurrent timing cannot be inferred from kernel counters.

**Alternative:** Application replay if state restore/progress fails, or separate source-equivalent device-counter diagnostic explicitly excluding CPU overlap. No silent dynamic/static partition change.

### sc

Host allocations from new int/std::atomic and device cudaMalloc/CuArray allocations are distinct and initialized via copies. Host threads run concurrently but do not respond through GPU flags. Inter-block device flag chaining is local to the device allocation.

**Required:** Restore all device data/flags for in-place compaction. Preserve alpha/cut and CPU/GPU concatenation/checks; full output counts/flags/guards mandatory. Device inter-block spin progress under profiling remains unproven for all instrumented schedules. Profiling cannot provide original CPU/GPU coexecution timing.

**Alternative:** Application replay if memory restore/progress fails; separately annotated kernel component measurement without claiming host/device performance equivalence.

### sss

Selected runtime uses two CUDA streams, asynchronous input/output observer copies and an explicit delete_ready event ordering. No same-allocation CPU atomic handshake was established, but full cross-stream read/write dependencies and whether kernel serialization changes the intended execution remain unproved by this bounded review.

**Required:** Review original retained host launch/copy lifetimes and both device bodies against exact event graph before multi-pass kernel replay. Require complete model/graph/raw-input/output/property checks. First-per-function/geometry omits vast evolving graph/search states.

**Alternative:** Reviewed application/range replay that preserves required stream ordering/concurrency; independent stream timeline before/after profiling. Keep full host/search time outside kernel roofline and do not use a small graph substitute as full search evidence.

## Additional cases requiring separate review

- allreduce/ccl/p2p/pingpong/ring: original two-device/communication contracts remain outside GPU6-only whole-contract execution. MPI/peer communication may require concurrent kernels/devices and host progress. Classify replay **unknown** until exact communication graph and profiler support are reviewed; no single-device rewrite.
- Additional full asynchronous word2vec workflow is not the selected primary CBOW component in this census. Twenty CPU workers,600callbacks/trial and asynchronous kernel queuing require their own source/config/ROI selection. Treat replay **unknown** for that additional workflow; do not inherit a CBOW component verdict or make equal-trained-model claims. Fresh application replay also needs state/output/host-scheduling/property audit.
- Retained CPU libraries or between-kernel host feedback alone do not prove kernel replay unsafe. Actual same-allocation live communication, concurrent kernel dependencies, and copy/launch lifetimes matter.

## Interpretation

No GPU/NCU or numerical experiment was executed. Safe rows require complete device memory restoration and all selected full scientific/output/guard checks to pass under profiling, with warnings/failures preserved. Spin-based kernels and nondeterministic convergence retain progress/scheduling caveats. Counters collected after serialization/cache flush are instrumented component data; they do not measure original host/controller time or concurrent stream throughput. First-per-function/geometry is not all algorithm stages/states/directions. No counter averaging or whole-benchmark compute/memory label is authorized by this report.

## Documentation provenance

Local NVIDIA ProfilingGuide (HPC SDK25.5) sections Kernel Replay/Application Replay/Range Replay were consulted and hashed in JSON. They are newer than active2022.4.1; available options and unsupported APIs must be checked against that actual executable before changing modes. The official2022.4 archive was inaccessible through browsing. No unsupported option is recommended as already validated.
