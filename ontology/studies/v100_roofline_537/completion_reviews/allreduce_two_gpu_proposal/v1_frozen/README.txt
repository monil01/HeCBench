ALLREDUCE ONLY: READY FOR ROOT CODE REVIEW; NOT EXECUTION AUTHORIZED

No GPU5 permission is implied by accepted historical two-rank results. Root must
review this actual code and record explicit additional user authorization for
physical GPUs5+6 as the FINAL step before --execute. No live authorization receipt
exists here. No worker launched the runner (even --help/default stage), compiler,
MPI job, NCU, CUDA query or GPU query. CPU source-copy builder only was executed.

Review entry points:
  proposal.json: all4 exact selected contracts/environment/input/source/tool hashes
  source_equivalence.json: native/common/entry inverse patches and exact callback prefix
  profile_window.h: host-only profiling/branch/count assertions; no math kernel changes
  run_proposal.py: finite locking/device/PID/LL/storage/check/counter acceptance
  rank_dispatch.py/rank_worker.py/session_entry.py: rank1-only NCU + actual worker identity
  check_artifact.py: independent complete bitwise input/output/received-temp/guards oracle

Exact proposed point:
  MPI rank1, logical CUDA1, physical GPU6, retained trial2, n1048576,
  first of3 ring repeats; offset0, count524288, 32blocks x256threads.
  MPI_Irecv/MPI_Send/MPI_Wait and context completion occur before profiler start.
  One finite FP32 add reads rank-private received tmp and its own output chunk.
  Profiling stops after completion and before final MPI_Sendrecv allgather.
  No MPI/NCCL/CCL communication is replayed. NCU kernel replay must restore written
  private output before additional passes; tmp remains read-only. Rank0 continues
  the unchanged accepted peer protocol, possibly waiting for rank1 at final gather.
  Source ordering is not actual runtime progress/tool validation.

Full computation stays intact:
  6 sizes1,32,33,256,4099,1048576; warmup1 ring/size/rank +5 trials x3 rings/size/rank.
  Host marker helper asserts96 authored ring-add calls/rank, rank1 exactly3 target
  candidates (trial2,size1M), enters/exits once; rank0 enters/exits0. All original
  full checks/read-only inputs/context fences and MPI PID guard remain unchanged.
  Native and all3 Julia implementations still check all5 full dual-rank artifacts.
  Runner independently checks every5,269,603 UInt32 payload words per trial bitwise:
  both actual inputs, both summed outputs, both received tmp chunks/tails, guards.
  Each trial21,078,412B; five105,392,060B. No numerical shortening/tolerance relaxation.

Guards:
  - Explicit --execute AND root-review/user-authorization receipt pinned to exact
    proposal+runner SHA, both UUIDs, all4 models and1800s/model are required.
  - Common GPU6 study lock plus explicit GPU5+6 pair lock, max60s acquisition.
  - Existing metadata UUIDs for physical5/6; actual both-device UUID/PCI/idle check
    before any build/profile, after build, and continuous1s foreign-use monitor.
  - Exact rank worker PID registered before exec/CUDA, UID/starttime/session/token
    and actual executable checked; no generic own-session GPU PID whitelist.
    Rank1 profiler has separate PID registration and may have GPU6 context only.
  - Registry/session mismatches or juliaup/NCU exec-fork incompatibility fail closed.
    Both rank workers must retain the fresh owned session. Cleanup only signals
    same-UID same-session generation-checked members; no other jobs are killed.
  - Child Landlock allows writes only to the fresh model slot (plus existing driver
    character devices/proc thread naming allowed by existing guard). Input/source
    packet/fixtures remain outside allowed write tree. No unrestricted fallback.
  - MPI temp, UCX POSIX backing, Julia/cache/TMP paths point inside fresh scratch.
    UCX_TLS/HPCX protocol unchanged. If those path adaptations are not supported,
    failure stays explicit; do not enable broad /dev/shm or alter transports.
  - Read-only source/config/result/dependency/code hashes checked before/after;
    compiled outputs hashed before profile and verified afterward. Actual effective
    rank environment and observed loaded DSO files are recorded prospectively.
  - Each model has1800s combined build/profile budget, max300s/build,120s registry
    startup,1GiB/file cap,512MiB enforced auxiliary quota and2GiB free reserve.
    Whole five-trial payload+aux+reserve required at dispatch. All files retained.
  - No measured point accepted without5 correct original rows +5 full independent
    artifacts +both final96-call guards +exact begin/end markers +one raw NCU group.
    Counter PID must equal authenticated rank1/marker PID; grid/block exact32x256;
    all PCI metric attributes must match runtime UUID-to-PCI mapping of physical6.

NCU rank filtering:
  mpirun still launches exactly2 ranks with accepted study_mpi_launch.py/HPCX env.
  rank0 execs accepted target without NCU; rank1 execs NCU around same target via
  registration wrapper. Installed2022.4 help supports application-only, --devices1,
  profile-from-start off, demangled regex, launch-count1, --kill0, kernel replay,
  cache-control all, clock-control none. --kill0 preserves all remaining trials.
  NVIDIA documentation describes --devices as visible CUDA device ordinals; with
  CUDA_VISIBLE_DEVICES=5,6 logical1 is expected physical6. Runtime PCI evidence is
  still mandatory; no zero-counter or wrong-device fallback is accepted.

Future CPU staging preview (not run here):
  python3 run_proposal.py --proposal /ABS/proposal.json --output-root /tmp/FRESH
Without --execute this only writes CPU directories/plans; no subprocesses.

Future authorized execution (NOT authorized now):
  python3 /ABS/run_proposal.py --proposal /ABS/proposal.json --output-root /tmp/FRESH \
    --execute --authorization /ABS/ROOT_RECORDED_USER_APPROVAL.json \
    --authorization-sha256 EXACT_SHA
Use absolute packet paths. authorization_template.json is explicitly false/null.

Interpretation:
  Raw counter diagnostic/local retained FP32-add point only. No full collective,
  rank0, other sizes, whole ROI, launch-count extrapolation or causal bottleneck.
  Instrumented STUDY_TRIAL times never replace/reinterpret accepted primary means.
  GPU5 is needed for the original scientific contract but receives no selected
  counters. No scalar-FP applicability or replay inference for ccl/NCCL/p2p/ring.
