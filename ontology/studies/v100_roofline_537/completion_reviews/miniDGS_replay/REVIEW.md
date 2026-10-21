# miniDGS: two MPI ranks on the same V100, kernel replay review

**All four selected model implementations: statically conditional safe for ordinary NCU kernel replay, with device-memory restoration enabled.** No source-level in-kernel dependency on CPU/MPI/another GPU process/another stream was found. This source conclusion is not a successful profiler execution, not authorization to bypass the write guard, not rank-specific ROI completeness, and not evidence that MPI overlap survives profiling. CPU-only review, no GPU/build/target test or source changes.

## Exact source/call order

`study_mpi.c:7–13` binds rank count exactly2 and carries `MPI_FLOAT` payloads in ordinary **host** pointers. `study_mpi_start` posts Isend(out) and Irecv(in) to1-rank with complementary tags6666+peer/rank. `study_mpi_finish` Waitall(in) and Waitall(out) before returning. There is no CUDA function, CUDA device pointer, mapped memory or GPU-aware MPI call in this CABI. `MPI_resolution.json` records OpenMPI4.1.6/libmpi.so.40.30.6; current library/source bytes and recorded identities are separately included in `per_model_review.json`.

Native `study_native.cu:14` allocates ordinary cudaMalloc buffers, preserving128wordguards each side; host send/recv are std::vector<float>. CUDA texture symbols installed atline27 reference only that process's context-local cudaMalloc arrays. Co-residence on physicalGPU6 does not alias allocations or CUDA context-global texture symbols across MPI processes.

Native selected driverline28 repeats exactly5RKstages:

1. partial_get_kernel3d<<<50,256>>> reads localQ/map and writes localtmp.
2. Blocking D2H cudaMemcpy(tmp→hostsend) completes that gather before MPI access.
3. CPU posts Isend(hostsend)/Irecv(hostrecv).
4. MaxwellsGPU_VOL_Kernel3D<<<511,84>>> reads localQ/DrDsDt/vgeo and writes localrhs. It **does not read hostrecv, devicepart or MPI state**.
5. CPU Waitall can overlap the volume kernel. It waits only on peer host MPI messages already posted after their gather, not on completion of a peer GPU kernel required by the local kernel.
6. After Waitall, H2D cudaMemcpy(hostrecv→localdevicepart) completes before the following source default-stream surface kernel.
7. MaxwellsGPU_SURF_Kernel3D<<<511,112>>> reads stable localQ/part/LIFT/surfinfo and updates rhs.
8. MaxwellsGPU_RK_Kernel3D<<<1150,256>>> reads rhs and updates res/Q. Next-stage gather is ordered on the same default stream.

Atline29, cudaDeviceSynchronize completes all work **before** the final MPI barrier, full-state CPU oracle/halo checks and rawcaptures. Texture destruction and cudaFree occur only after completion. Native conditional warmup captures also force D2H completion between stage observers. All source kernels are finite: volume loops boundedp_Np84, surfacep_Nfaces*p_Nfp112, gather12768, RK294336. Only intra-workgroup barriers occur; there is no device polling flag, persistent CPU-fed queue, system atomic rendezvous or remote-memory progress dependency.

Julia shared `study_common.jl:63–114` implements the same order:

- `make_buffer` uses rank-local ordinary CuArray/backend arrays; send/recv are Vector{Float32}, protected byGC.@preserve throughMPI completion.
- line75 run_gather, copyto! into hostsend, then CUDA.device_synchronize beforeMPIstart.
- line76 starts host-onlyMPI; line78 launches volume on localQ/D/G/rhs.
- line82 waits both MPIrequests, then copies hostrecv to devicepart. Surface/RK follow in the same current task's stream.
- line96 device_synchronize before time stop and finalbarrier; full checks and frees after that.
- Send is not reused and recv is not uploaded until both MPI requests have completed. MPI may mutate recv while volume is active, but that memory is host-only and not an input of the volume kernel.

## Per-model backend ordering

| Model | Source launch order / memory | Device synchronization |
|---|---|---|
| NativeCUDA | Source defaultstream, context-local texture handles and cudaMalloc buffers | Blocking gathered D2H and received H2D; finalcudaDeviceSynchronize; ordinary same-stream dependence betweenvolume/surface/RK |
| CUDA.jl | @cuda currenttaskstream, localCuArray buffers | Gather copy→device_synchronize; source same-task launch/copy ordering; finaldevice_synchronize |
| KA | CUDABackend workgroups256/84/112/256, exact whole ndranges | Same shared harness completion points; finite localworkgroup barriers; no cross-rank barrier in devicecode |
| JACC1.4.0 | Explicit launch shapes and shmem0/2352/2688/0,sync=false | LaunchSpec default_stream maps directlytoCUDA.stream(); CUDAExt launches withspec.stream; same shared harness completion points |

All kernels/parameters remain exactly those in the selected configuration and batch023 plans. JACC zero-shared templates are ordinary stable rank-local device arrays, not host concurrency flags. The backend implementation files and pinned environment manifests are source-bound in the machine-readable review.

## Why serialized profiling does not create a source dependency cycle

A rank stalled before a gather/volume launch does not hold an executing GPU kernel that awaits a peer message. An instrumented gather/volume/surface/RK is a finite device computation. After that launch completes, the rank can post or finish its CPU communication. The other rank can then acquire the profiler device resource and advance. CPU Waitall may be delayed, and MPI/GPU overlap may disappear, but no replayed kernel requires an un-replayed CPU action on every counter pass.

NCU must restore mutable **rhs/Q/res/tmp/part and any other written allocation** for each kernel pass. Surface accumulates into rhs, and RK updates res/Q in place. Repeating them without restoration would change the trajectory; a no-save shortcut is not permitted. Ordinary kernel replay with memory checkpoint/restore is the reviewed method, not application replay and not a microkernel with synthetic state. CPU Isend/Irecv need not repeat for each counter pass because the currently consumed device haloQ is frozen and rank-local during its surface kernel.

## Conditions and remaining unknowns

1. Preserve mpiexec-n2 and fixed physicalGPU6namespace/alloriginalshapes/all5checkedtrials. Do not rewrite to one rank or remove communication.
2. NCU must follow both processes (`--target-processes all`) and use the owned shared profiler TMPDIR/lock strategy. Actual OpenMPI transport/NCU injection compatibility under the childwriteguard remains an execution gate. Source safety does not authorize unrelated filesystem writes or foreign locks.
3. Preserve strict full numerical/guard/halo checks and per-ranktrace persistence. A failed instrumented run is retained as failed, even if some counters were produced.
4. Counter replay serializes rank work and does not measure source MPI/GPU overlap, networkprogresscosts, or the primary max-rankwalltime. No causal throughput comparison may substitute sampledkernel durations for the original full pipeline.
5. Firstmatching kernel specialization/geometry may be warmup, setup or one RKstage. It does not establish all5stages, bothranks, alltrial coverage or a wholebenchmark roofline.
6. Kernel rows record PID/context; current rank logs do not explicitly pair OS PID with MPIrank. Rank-specific counter attribution remains unknown unless a separately reviewed host-only PID/rank observation is captured. Do not infer rank from connection order or assume exactly8rows: backend/JIT specialization/setup kernels and crossprocessfilter accounting may differ.
7. Pure application replay would need deterministic two-rank launch/stream matching and a separately reviewed launcher policy. It is not necessary to address any source in-kernel host-progress dependency found here and is not automatically authorized.

No mathematical model/source/kernel changes are proposed. Scope is only a source-level replay progress resolution for these exact selected model implementations; sampled phase/resource differences remain covariates.
