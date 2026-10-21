# A–M selected profiling completion review

CPU-only source/saved-evidence review, no GPU/build/numerical benchmark runs, shared tooling/ledger writes, accepted-source edits, foreign-process intervention, sudo or Git. One newly authorized recovery tool was added; all broader isolation copies are under this private directory and remain **source packets, not executable plans**. Actual observed usage is shared profiling review, not translation cost.

## Selected snapshot

`selected_failure_scan.json` binds each selected parent provenance, plan, counters/log/build files and ledger snapshot. At the scan,207 selected A–M slots were not completed:

| Category | Slots | Action |
|---|---:|---|
| Foreign contention |133|Fresh ordinaryv7 plans after fixed GPU idle/recheck; never stop foreign processes. Exact list `foreign_contention_retry_slots.json`. Parent retry021 now exists; do not duplicate it.|
| Operational deadline |13|bitcrackerJACC1,columnarSolverJulia3,CRSKA1,dispatch4,LSQT4. Bounded scoped sampling or reviewed larger operational budgets; retain failed processes/partial counters.|
| Native hardware barriers |16|Four Blackwell-era BLAS cases×4models; documented native barrier, no arithmetic substitute.|
| Replay-method hold |4|atomicSystemWide CPU/GPU concurrent atomics on same managed memory. Existing kernel replay cannot reproduce host actions; needs separately reviewed method.|
| Plan gates |15|allreduce/ccl8 are original multi-device contracts; BNnative,clocknative,GC4,memcpynative7 are precise input/output role/isolation issues.|
| Successful process / no CSV header |7|mallocFree4 and memcpyJulia3 are source/runtime-only; kernel roofline not applicable, not fabricated zero metrics.|
| Other |19|ASSERTJulia3; bitcracker3 provider failures; CCSnative tmpfile; CMJACC historical trace; columnarSolvernative missing compiler path; EGS4 intercepted runtime compiler; HeartwallKA CSV field length; MF-SGDnative missing fresh outputdir; miniDGS4 historical persistence writes.|

Counts are snapshots of selected assessment slots, not final current campaign completion. Later retries or refreshes must supersede them explicitly. No newer successful raw attempt was found for these selected slots at the duplicate-work check. The root controller and refresh processes are left untouched.

## Immediate CPU-only completion: Heartwall KA

New source-owned helper: `tools/v100_roofline_recover_counter_derivation.py`.
Actual saved-counter recovery completed in `batch022-heartwall-counter-recovery/heartwall/ka` under raw-root. Original target processrc0, no stop reason/foreign activity, captured after-binding recheck unchanged. The sole kernel name is196,542characters, exceeding the former131,072CSV limit. Recovered1row has positive duration, non-null DRAM traffic, exact savedPCI(0,203,0) matching parent UUID mapping. NewGPUprocesses0/buildprocesses0/countersamples0. All samples are the independently retained original row.

The tool retains parentplan/counters/rawlog/buildlog/collector/guard/auxmanifest bytes; `parent_profile_provenance.json` and `parent_parser_source.py` preserve old identities. It freezes new parser and tool source, calibration snapshot, `counter_derivation_recovery.json`, derived data and new provenance. Historical target source identity comes solely from parent captured bindings/checks; today's source hashes do not certify that past execution. The independently measured half calibration is separately bound to its V100UUID/SM70 proof, with tensor conversion disabled. Root should retain all recovery metadata when freezing. Full ROI/warmup/source candidate qualifications remain unchanged.

`heartwall_ka_cpu_reparse.json` is the earlier private proposal derivative; it is **not** the new accepted campaign receipt. Use batch022 for integration.

## Terminal kernel roofline applicability: seven runtime cases

`runtime_no_kernel_applicability.json` supplies each exact parent profile/plan/counter/log/result/config and source receipts. All reviewed current bytes match the original captured supplemental source SHAs. Each of seven models has actual five checked correct rows, successful uncontended process and explicit NCU `No kernels were profiled` warning.

* mallocFree4 uses the same narrow CABI cudaMallocManaged/cudaMalloc/cudaHostAlloc/cudaFree/cudaFreeHost. Each trial checks28,500allocation/free pairs with statuses/non-null/lifecycle, across57type/size cases. No source device kernel or numeric output exists. Source execution-only status stays separate.
* memcpyJulia3 invokes CUDA.jl copyto!, KA copyto!(CUDABackend) or JACC.transfer! on host/device Int32 arrays. Full actual payload/guard/readback numerical checks remain valid. These observed transfer-engine executions have no sampled SMkernel, so a kernel FLOP/DRAM roofline is inapplicable. Numerical memcpy agreement is **not** converted into an execution-only benchmark.

Propose source-qualified terminal statuses `roofline_not_applicable_no_device_kernels` and `roofline_not_applicable_runtime_transfer_engine_no_profiled_kernels`. Metrics remain null, not0FLOPs/bytes. A different API/copy-engine timing/memory-tier study could be useful, but none is invented here. No GPU retry is needed to recover absent kernel counters.

## Concrete fresh retry / isolation proposals

`non_source_retry_proposals.json` and `source_overlay_proposals.json` bind exact sources and selected parent contracts. All old outputs remain protected.

* **columnarSolver native**: `/usr/local/cuda/bin/nvcc` does not exist. `/usr/bin/nvcc` is the existing59B wrapper executing `/usr/lib/nvidia-cuda-toolkit/bin/nvcc`; both SHA-bound. Actual `--version` reports CUDA12.0/V12.0.140, not a new toolchain. Replace only fresh build executable, keep flags/source/inputs and write guard. No compiler/build ran here.
* **MF-SGD native**: profile argv[4] names fresh `profile_auxiliary`, but source opens `argv[4]/cuda_pid_trial_case.*` without creating it. Precreate precisely this owned argument directory before running; no source patch or removed scientific gate.
* **CCS native**: C `tmpfile()` fails under Landlock because libc's fixed temporary location is outside the allowed tree. Proposed copied-host adapter uses exclusive mkstemp within the new scratch, immediate unlink and fdopen(w+b); source printbicluster/seek/complete printed bytes remain. Parent compile validation is still required; no build occurred here.
* **CM JACC**: copied child changes only hardcoded historical scratch-initial/final trace root to fresh output/cm_trace; all device functions and ordinary includes retained, canonical includes pinned.
* **miniDGS4**: copied Python wrapper keeps canonical source P for fixtures, plans and original2MPI-rank child commands, changes only historical retained rank persistence to `out/retained_rank_outputs`. All raw ranks/full states/independent checks remain. No copied-wrapper execution occurred.
* **BN native**: even its failure-only warmup dump must be writable under the guard. Redirect only that dump; do not remove the failure branch. FIXTURE_ROOT remains canonical read-only.
* **clock native**: fixture path is canonical read-only input; raw timing sidecar root changes to fresh output. Deviceclock/resource measurements remain subject to existing interpretation limits.
* **GC4**: both native and Julia common raw-sidecar roots need fresh output. Julia copied child must include copied common; helper currently produces source packet only and flags this required final plan step explicitly.
* **memcpy native**: pattern fixture is canonical read-only; change its fixed `/tmp/memcpy_pid_trial.bin` outputs to fresh output. Future no-kernel applicability must follow actual successful execution, not guessed from the other three.
* **EGS4**: original failure is the compiler subprocess invoked *inside NCU target-processes-all*, exit255 with no diagnostic localization. Prebuild unchanged `check_mt_trajectory.cpp` with original-O3/-std=c++17 into fresh scratch mtcheck before NCU. Julia finalizer already skips compilation if it exists. Native copied wrapper can require the reviewed prebuilt checker instead of recompiling; complete independent physics/trajectory checks still run. Exact root cause of compiler255 is not proven, and no build was run here.
* **bitcracker4**: independent Python cryptography AESCCM oracle fails to load the OpenSSL3 legacy provider under the profile's loader environment. Source uses AESCCM, not a legacy cipher. A dependency-only `CRYPTOGRAPHY_OPENSSL_NO_LEGACY=1` proposal requires CPU AESCCM dependency validation under the same loader paths before retry; not performed here. Do not remove the independent check or accept correct:false. JACC also exhausted its~690s deadline.
* **ASSERTJulia3**: previously reviewed positive-only profiling proposal already exists in `study/assert_review`. Preserve negative diagnostic failure/unknown cause and execution-only source meaning; no duplicate new helper.

`prepare_source_overlays.py` is a private no-execution source-packet helper. It verifies exact original SHAs and replacement cardinalities, refuses an existing/outside-/tmp destination, copies only reviewed host output changes and canonical read-only includes, records full diff/hash and source-before/after. Three packets were prepared (CCS/CM/miniDGS), **none is a runnable plan or permission to clear unrelated gates**. Other proposal cases are regenerable, not staged. No math/kernelbody/geometry/tolerance/iteration change is authorized by this helper.

## Timeouts, genuine limitations and progress

ColumnarSolverJulia3 already spent1273–1300s per attempt while only1kernelcounterrow was selected. That evidence does not establish a hang or correct outcome under profiler; inspect source's long-work/replay/synchronization state before simply increasing deadlines. Dispatch4 exhausted180s with1sample while original source intentionally performs many no-work launches; profile-firstsample does not remove source host launch workload, so an operational deadline from per-launch normalized time is unsuitable. It remains execution-only, never numerical roofline proof. CRS KA captured50shapes by180s, and LSQT4 captured43kernels while CPU/GPU DOS/VAC/MSD steps visibly progressed: these are bounded-budget issues, not missing APIs or numerical defects. Parent retry021 is now responsible for selected timeouts/contention; no duplicate process was launched.

allreduce/ccl retain source multi-device requirements; no silent GPU6single-device replacement. Four reviewed BLAS hardware barriers and atomicSystemWide replay hold retain their existing evidence/qualifications. Neither a blocked profile nor an exhausted operational budget establishes framework unsupported. Full elapsed-ROI aggregation and causal bottleneck claims still require source-counted coverage, launch multiplicities, memory tiers and calibration; successful partial counters alone do not supply them.
