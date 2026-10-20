# Workflow replay assessment

CPU-only source review of all12 selected model slots, including source hostMPI/GPUcollective/data pointer lifetimes. Safe is conditional static per-kernel state/progress proof only; tool/platform/transport integration and full output checks remain mandatory.

| Workload | Models | Verdict |
|---|---:|---|
|Sparkler|4|conditional static safe|
|SLU|4|conditional static safe|
|SSSP|4|conditional static safe|

## sparkler

All4MPI ranks are distinct processes/contexts, with rank-local cudaMalloc/CuArray A/B/C and independent cuBLAS handles/streams. GEMM uses only those local device pointers; no device pointer is passed to MPI. Barriers precede GEMM only after stream completion and follow it after stream+device completion. Allreduce/Gather communicate host scalar/checksum/byte buffers outside running GPU kernels. No selected device kernel waits for another rank/host collective response, so serializing finite kernel launches introduces no source progress cycle.

### Memory/progress model

- **rank_local_input_prep:** Writes own A/B only, completes prep event before use.
- **rank_local_GEMM:** Reads own A/B, writes own C, no remote-device/host progress dependency.
- **collectives:** MPI_Bcast nonce; MPI_Barrier after GPU completion; Allreduce/Gather host scalars/checksum/character arrays; no running-kernel memory handshake.

### Conditions

- All4ranks must remain launched, on physicalGPU6, with --target-processes all and one shared private TMPDIR for profiler interprocess serialization. Do not profile rank0alone and claim all-rank coverage.
- Restore mutated C/vendor workspace and prep buffers between passes; preserve rank-specific active/inactive steps, rank0/1threeGEMMs versus rank2/3twoGEMMs, source streams and host collectives.
- Retain every rank CAS/ledger/full actual matrices/guards, independent matrix arithmetic and final gathered complete output checks; no checksum-only acceptance.
- NCU may serialize independent rank contexts, alter barrier wait/CPU scheduling and remove actual multi-process overlap. Instrumented walltime and sum of per-kernel counters are not original MPI4 throughput or the full45interval ROI.
- Actual profiler/MPI transport/injection compatibility and private temporary file behavior still require observed checks. Landlock failure, missing rank, timeout or skipped data remains a failure, never permission to weaken collectives.

**Alternative:** Conditional per-kernel replay candidate for rank-local prep/vendor-GEMM counters. Use a separate timeline/concurrency-aware method for MPI4overlap and rank wait. Application replay requires deterministic rank/context/kernel matching and fresh output roots; no auto switch.

## slu

Selected numeric CSC/CSR/value/level/workspace arrays are cudaMalloc device allocations, uploaded before numeric work. CPU NICSLU symbolic preprocessing/leveling precedes LUonDevice; CPU solve reads factors only after final completion/D2H. All six native entries launch in the ordered default stream. Julia sets legacy stream explicitly; JACC specs bind legacy stream, KA uses the current legacy task stream. Nstreams=16 is a workspace slot/host batching constant, not16CUDAstreams. There is no live CPU producer or other-stream GPU kernel required to release a running kernel.

### Memory/progress model

- **GPU:** Reads sparse structural indices/levels; modifies numeric values and tmp workspace; kernels ordered on legacy/default stream.
- **CPU:** Symbolic analysis/leveling before uploads; host-level schedule uses fixed symbolic metadata; solve/physicalLUchecks after factor D2H.
- **Nstreams:** 16integer workspace slots, no corresponding cudaStreamCreate in selected integrated numeric path.

### Conditions

- Restore the entire mutable factor val_dev AND tmpMem state before every replay pass, not merely final C/output. Workspace is10353*nFloat32 words (~4.11GB for fullASIC), written and cleared during numeric work even though outside-ROI captures are allzero.
- Factorize→update→clear kernel ordering and between-level dependency must remain exact. CPU solves and independent fullLUproduct/system residual checks must still pass; preserve all immutable indices and before/after workspace checks.
- Float32 atomic updates may vary summation order. Apply original declared numerical/backward-error contract, not fabricated exact factor bytes or relaxed tolerances.
- Checkpoint overhead/device-or-host backup memory pressure may be large; budget active device/host memory and preserve any profiler failure. It does not justify reducing the source problem or omitting workspace restoration.
- First each name/geometry does not represent all48kdispatches, level/column states or six entry variants. CPU NICSLU preprocessing/solve is outside original numeric ROI and must not contribute invented GPU FLOPs/bytes.

**Alternative:** Conditional per-kernel replay with complete factor/workspace restore. If memory pressure/restoration fails, review fresh application replay preserving full factorization/sourceCPUcontroller and original mathematical contract; noGPU-only substitute.

## sssp

The selected wrapper alternates CPU and GPU frontier phases. CPU source worker call joins before return; a CPU phase completes before host state is uploaded for GPU work. GPU phase uses separate cudaMalloc/CuArray cost/color/q1/q2/sc buffers and completes before scalar D2H/host frontier decision. Each GPU launch handles a finite snapshotted input frontier with device-local atomic head/tail; no host worker feeds the running kernel.

### Memory/progress model

- **CPUphase:** Operates host vectors only; outer worker std::thread joins before hostlevel returns.
- **handoff_to_GPU:** Cost/color/q1/q2/sc H2D, contextsync; scalar reset upload before each launch.
- **GPUphase:** Device-only finite frontier, atomichead/tail and relaxation updates; contextsync then scD2H/hostdecision and final stateD2H.

### Conditions

- Restore ALL mutated device cost/color/outputqueue/head/tail/overflow/iteration state before extra passes. Otherwise later passes consume an advanced head/frontier or accumulated relaxations and are not the original kernel instance.
- Keep snapshot nt/gray shade, queue parity, threshold128 and120blocks. CPU/GPU handoff is serial and explicit; do not replace it by a GPU-only traversal or a live CPU/GPU shared frontier.
- Frontier order is nondeterministic; preserve original fullInt32cost-vs-independent-Dijkstra, actual graph inputs, queue/scalar/redzone/overflow validation and original raw-queue-identity limitation. Do not require or manufacture equal raw queue order.
- First function/geometry selects only one evolving frontier despite same grid acrosslevels. Sourceprimary time sums CPU+completedGPUsegments, excludes transfers/control/observers; profile kernel counters cannot supply fullCPU/GPUalgorithm FLOPs or pipeline bandwidth.
- No managed/mapped host allocation backs selected GPU frontier arrays; do not infer AtomicSystemWide semantics from std::atomic in the CPU helper.

**Alternative:** Conditional per-kernel replay with complete frontier snapshot restore. If source progress or independent final cost checks fail, preserve evidence and review application replay retaining allCPUlevels/handoffs; select exact frontier state explicitly before any fullworkload aggregate.

## Source vs profiler behavior

Sparkler has no source progress cycle when independent rank-local kernels are serialized: all MPI waits surround completed local GPU work, and no active kernel consumes remote/hostMPIdata. This does not prove any particular profiler/MPI transport setup will run successfully. The newer local NVIDIA CLI guide documents all-rank profiling with target-processes all; active2022.4 compatibility still needs actual process/rank/counter/output evidence. The same private TMPDIR must reach all ranks so profiler serialization does not use unrelated lock files. Never modify another user lock or widen source/history write access.

Device memory restoration is a required part of each safe assessment. In-place factor/frontier updates are not idempotent: accepting a successful launch without restoring prepass state would be wrong. No fixed iteration, single-source-only kernel or changed work partition is proposed as a substitute. Full independent checks remain mandatory after profiling; original immutable five-run runtime evidence is untouched. Counter samples under serialization/cold cache are kernel-component observations, not causal whole-application bottleneck claims.

No compiler, GPU/NCU, numerical or benchmark execution. Reports/sourcehash checks only. No source/shared metadata/Git changes.
