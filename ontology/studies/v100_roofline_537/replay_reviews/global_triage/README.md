# Cross-suite replay triage

CPU-only review reads all seven source cohorts plus initial nine (532 accepted names), retains candidate search evidence, and manually reviews 27 selected controllers across four models. `review.json` contains 340 source/config/result-bound candidate or prior-review records and an explicit 532-name index. Names without a focused finding remain **unknown**. All current bound sources and result files still match their cohort snapshots. No GPU query, profile, build, Git, sudo or shared benchmark/measurement edit occurred.

**The safe status means only that the selected isolated kernel has no established live external-progress dependency. It is not profiler-tested approval, wholeworkflow replay fidelity, race freedom, memory-residency equivalence, complete ROI evidence or causal diagnosis.** Every record has `profiling_authorized_by_this_artifact=false`.

## Findings that need action

| Selected workload | Finding | Default isolated replay classification |
|---|---|---|
| ccl | two-rank NCCL AllReduce requires matching peer/proxy progress | unsafe |
| pingpong | selected NCCL send/recv branch requires peer/proxy; MPI GPU-buffer branch transport remains unknown | unsafe for NCCL; no MPI blanket approval |
| allreduce | CUDA-aware MPI ring between fenced local reductions | unknown whole transport/process protocol |
| miniDGS | two MPI ranks on GPU6, host halo exchange overlaps local volume work | unknown multi-process replay; no direct peer address spin inside authored kernel established |
| sparkler | four MPI ranks on GPU6, local GEMMs bracketed by barriers | unknown multi-process replay; serialized samples cannot preserve rank/barrier timing |
| halo-finder | all four models directly compute on mapped pinned-host/UVA arrays | unknown exact replay restoration; host/PCIe memory-tier qualification required |
| nbnxm | all four share managed allocations, GPU prefetched before accumulated force loops | unknown exact restore/residency preservation |
| snicit | managed arrays read/updated by CPU only after each stream-synchronized callback | no live handshake established; managed state/phase replay remains unknown |
| blas-gemmEx | native uses managed + prefetch; Julia models use framework device arrays | per-model memory policy; no uniform managed claim |
| concurrentKernels, streamPriority | finite kernels with event dependencies; source-admissible isolated progress | concurrency/workflow timing not preserved |
| overlap, perlin, lfib4 | finite kernels, independent buffers/strips or explicit stream joins | isolated progress admissible; original overlap not preserved |
| nosync | host future Thrust reduce on legacy default stream plus custom-stream scan | unknown vendor/thread/stream replay protocol |
| lda | per-word spinlocks between blocks inside one kernel | unknown instrumented progress; not a live CPU/other-kernel handshake |

The selected `rsc` fitting worker threads join before parameter H2D and GPU evaluation. Selected `sssp` explicitly alternates distinct host/device states via copies and fences; its internal GPU progress remains unknown. `segsort` host bin feedback follows an event-completed D2H. `simpleMultiDevice` selected argv is **g1**, not a multi-GPU experiment. `miniWeather` and HPL enforce single MPI rank. `minisweep` selected finite wavefront loops have nproc1/nblockz1; no separate persistent producer/consumer kernel is established. `threadfence` elects a last-finishing block and resets its counter, rather than spinning on a host response. Runtime-only streamCreateCopyDestroy and mallocFree have no authored replay kernel.

Already-reviewed atomicSystemWide/streamUM/sss/tqs/winograd/zerocopy/hungarian/openmp/prefetch/blas-gemmEx2/ans/bsw/base64e/pad/sc are deferred explicitly; this artifact does not weaken their prior restrictions.

## Halo-finder: exact all-model pointer proof

`halo_finder_all_models.json` separately binds selected result/config and all recorded source hashes for CUDA, CUDA.jl, KA and JACC. Shared `halo_dispatch.h` allocates pinned host payloads with cudaMallocHost. Configured RCBForceTree allocates force workspaces through that wrapper and passes the exact eleven fields `x,y,z,mass,x1,y1,z1,mass1,vx,vy,vz` in HaloArgs. The shared observer checks cudaMemoryTypeHost and devicePointer==hostPointer before and after each scientific force callback.

The native null callback invokes Step10_kernel on those addresses. The three Julia callbacks use shared `halo_wrap`: unsafe_wrap(CuArray,CuPtr(UInt(p)),own=false) on the exact same eleven pointers, then dispatch each model's authored force kernel. This is an alias; it does not allocate/copy HBM storage. All relevant controller/header/configured source and Julia callback hashes match selected records.

The output fields vx/vy/vz receive atomic additions. Their original mapped-host bytes must be restored before additional replay passes; otherwise the additive computation would repeat on changed values. The observer requires incoming velocity zero for each force call. Source fences/observer synchronization bracket the callback; no simultaneous host write inside the force kernel is established.

The installed Nsight Compute **v2022.4.1** guide explicitly describes saving GPU-accessible memory, restoring the written subset between passes, and a strategy for host-origin allocations. Therefore **hostmapped restore is not declared unsupported**. Exact target restore correctness has not been observed in this CPU-only review. Retain raw counters and qualified L2 projections; an HBM-only DRAM-bound envelope does not describe the mapped-host primary operands.

## Method constraints

`installed_profiler_guidance.json` hashes and excerpts the installed primary guide. It documents kernel serialization within and across processes, a profiler lock, the hang risk for kernels needing host responses during replay, and supported range replay constraints. Application replay still needs deterministic launch/context/stream/rank matching; it is not a generic fix for mandatory concurrent kernels. A reduced single-pass metric set can avoid memory restore but may still serialize peer kernels. A range/full-graph or coordinated application method requires a separate API/lifetime/complete dependency review and validation.

Pattern scans include unused original code, headers and controls, and exclude packages/files over2MB; complete exclusions are retained. No-match is never interpreted as safe. These findings preserve selected execution/property/numerical qualifications and primary timing means. There is no new fullROI or wholebenchmark bottleneck claim.
