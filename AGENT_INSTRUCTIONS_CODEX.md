# HeCBench Julia Porting — Agent Instructions (Codex)

> **Which file to read.** Two run-books live at the repo root:
> * `AGENT_INSTRUCTIONS_CODEX.md` — this file. Follow it when the
>   driver is OpenAI Codex (CLI or API).
> * `AGENT_INSTRUCTIONS_CLAUDE.md` — companion for Claude Code.
>
> This Codex run-book is intentionally scoped to **Julia (`CUDA.jl`) ports
> only**. Do not create, repair, or extend Serial, Triton, Mojo, Rust, HIP,
> SYCL, or OpenMP ports unless the user explicitly changes the scope.

This document is the run-book for the **next Codex-driven agent** that
continues the HeCBench Julia porting effort. Read it
end-to-end before doing anything else — the "why" for every step is
captured, so you can make judgement calls when reality diverges from
the recipe. After each completed Julia port, write a local context
checkpoint under `${STATE}/porting_logs/<bench>-julia/` before starting
another port. Do **not** ask the user to run a remote/platform context
compression task.

## Codex context hygiene — avoid remote compaction

Codex remote context compaction has been observed to fail with
`404 Not Found` from `/backend-api/codex/responses/compact`. Treat remote
compaction as unavailable for this project. The run-book must be executable
from local files plus git history, not from a long chat transcript.

Operational rules:

* Default to one Julia port per agent, but when the user explicitly asks for
  parallel agent work, split the remaining worklist across sub-agents. Start
  with the verified short-list (§3.2.2), then continue into Batch A/B/C
  candidates when that list is exhausted so the run can reach 50 more Julia
  ports. Each sub-agent must own disjoint `src/<bench>-julia/` directories
  and must still finish, verify, test, log, track, and checkpoint each port
  before moving to its next assigned port. The current batch objective is
  for each agent run to port at least 50 remaining benchmarks and at most 200
  benchmarks before moving to broader Batch C work.
* Before reading or pasting large outputs, write them to
  `${STATE}/logs/*.log` and summarize only the decisive lines in the chat.
* Do not paste full source files, full build logs, or full verification logs
  into the conversation unless a specific short excerpt is needed.
* After every successful port commit, write the local checkpoint described in
  §6. In a parallel batch, the coordinator may keep the session open while
  sub-agents work, but each completed benchmark still needs its own checkpoint
  before another benchmark is assigned to that same worker.
* If context is getting long before the port is done, pause implementation
  long enough to write
  `${STATE}/porting_logs/<bench>-julia/context_checkpoint_inflight.md` with
  current files changed, commands run, failing error, and the next concrete
  step. Continue from that file instead of requesting remote compaction.
* If a Codex UI or CLI offers remote compaction, do not trigger it manually
  for this workflow. Use local checkpoints and fresh sessions.

## 0. Codex restart scope — keep the existing ports

Every `src/*-julia/` directory that already exists on this branch may have
been produced by another agent. **Do not re-do it, do not delete it, and do
not overwrite its manifest.** Your job starts at the *worklist gap* — the
CUDA benchmarks that still have no Julia sibling.

Concretely:

* Read the "already done" lists produced by §3.1 and *skip* those
  benchmarks unless the user explicitly asks for a re-port.
* If you re-run the verification on an existing Claude-authored port
  (fine, encouraged as a checkpoint) and it fails, record the failure
  under `${STATE}/porting_logs/<bench>-julia/reverify_YYYY-MM-DD.md`
  and flag it — don't rewrite the port until the user says so.
* Every port you *do* produce gets `"driver": "codex"` in its
  `manifest.json` (§6). This is what lets §7's cost-study tables
  keep the two drivers in separate columns.
* After each completed Julia port, write a local checkpoint before starting
  the next port. Use
  `${STATE}/porting_logs/<bench>-julia/context_checkpoint_NN.md`.
  Summarize the commit, files changed, verification commands/results,
  manifest path, worklist updates, remaining unrelated worktree changes, and
  any pitfalls the next agent needs. Do not request remote context
  compression from the user.

**Current focus (this pass): make the Julia (`CUDA.jl`) ports numerically
correct against the CUDA references.** The broad translation tranche is
finished/triaged:
the live tree has 537 CUDA benchmark directories, 448 Julia benchmark
directories, 89 CUDA benchmarks without Julia directories, 91 local
deferred records, and 0 actionable non-deferred missing benchmarks. The
coverage database currently records 441 Julia `ok` results and 5 Julia
`run_fail` results (`aobench`, `kalman`, `lrn`, `mis`, `tsp`). Do not
continue broad translation work unless the user explicitly reopens a
specific deferred or failing benchmark. The active project objective is to
correct the failed numerical checks and close the gaps from the numerical
accuracy sweep in §11.


Existing non-Julia implementations may be used as reference material or
verification inputs when they already exist, but they are not targets for new
work in this pass.

The paper that consumes these ports is being written in
`Hecbench-agent-paper/IEEEtran/` (sections currently empty scaffolding); the
data flowing into it — verification status, execution time, iteration count,
token count, error taxonomy — is produced by the workflow described below.

---

## 1. Goal in one paragraph

HeCBench ships kernels in CUDA / HIP / SYCL / OpenMP-target. The broad
Codex Julia (`CUDA.jl`) translation tranche is complete for this pass:
passing ports were committed, infeasible remaining CUDA benchmarks were
recorded as deferred, and no actionable non-deferred missing Julia
benchmarks remain. The active goal is now to audit numerical accuracy for
the existing Julia ports by comparing CUDA and Julia outputs more strongly
than the current structural `tools/verify_coverage.py` fallback. The paper's
tables and figures should consume both the existing coverage/porting records
and the new numerical-accuracy records described in §11.

---

## 2. Current state — what already exists

### 2.0 The `${STATE}` directory (Codex-specific)

Codex has no session-scoped scratchpad the way Claude Code does, so
this run-book uses a **fixed on-disk state directory**. Every command
in this file that reads or writes `${STATE}/...` refers to that path.

Set it once at the top of your session, then re-export it in any new
shell:

```bash
export STATE="${PORTING_STATE_DIR:-$PWD/.porting-state}"
mkdir -p "$STATE"/{logs,porting_logs}
```

`.porting-state/` is already covered by `.gitignore` (added in this
commit); do not stage anything under it. Contents you write there:

* `${STATE}/coverage.db` — SQLite from `verify_coverage.py`.
* `${STATE}/perf.csv` — CSV from `perf_sweep.py`.
* `${STATE}/logs/*.log` — build/run stdout+stderr.
* `${STATE}/porting_logs/<bench>-julia/` — per-port prompt +
  response + error + manifest (§6).
* `${STATE}/porting_logs/<bench>-julia/context_checkpoint_NN.md` —
  local handoff summary written after each completed port instead of asking
  the user for remote/platform context compression.
* `${STATE}/{cuda,omp,julia}.txt` — worklist snapshots (§3.1).
* `${STATE}/julia_missing.txt`, `${STATE}/batch_A.txt`,
  `${STATE}/batch_B.txt`, `${STATE}/julia_deferred.txt` — Julia worklists.

Two tools currently hard-code an old Claude-session scratchpad path:
`tools/verify_coverage.py` (`DB_PATH`, `log_dir`) and
`tools/perf_sweep.py` (`OUT_CSV`). Before your first run, patch both
to read `${STATE}` from the environment — a two-line change per tool.

### 2.1 What already exists in the repo

Ground truth is the file tree, not this document; verify before trusting.

* Existing Julia ports live in `src/*-julia/`. Treat those directories as
  already done unless the user explicitly asks for rework.
* Other target-language siblings may exist and can be read for reference, but
  they are not part of this Codex pass.
* **Reference sources** for every benchmark live in `src/<bench>-cuda/`,
  `-hip`, `-sycl`, `-omp` — the CUDA one is authoritative for the port.
* **Tooling** (all in `tools/`):
  * `verify_coverage.py` — builds a benchmark for a list of models, runs
    each, and cross-compares stdout with a canonicalizer that (i) strips
    timings, (ii) trusts the bench's own `PASS`/`FAIL`, (iii) falls back to
    structural comparison of numeric summary lines. Writes to a SQLite
    `coverage.db` in the session scratchpad.
  * `perf_sweep.py` — grid over `(bench, model, size)`, runs each cell with
    a per-model timeout, parses the "Average execution time" line, appends
    to `perf.csv`.
  * `perf_sweep_to_tex.py` — turns `perf.csv` into LaTeX rows for
    `Hecbench-agent-paper/IEEEtran/4.perfomrance_comparison.tex`.
  * `perf_plot.py`, `perf_plot_scaling.py` — matplotlib plots.
  * `hecbench` (CLI) — repo-wide list/build/run wrapper on the CMake tree.
* **Language environments** already set up:
  * Julia: shared project at `src/_julia_env/` with `CUDA.jl` +
    `StaticArrays`. Every `-julia/Makefile` passes
    `--project=/home/imo/HeCBench/src/_julia_env`.
  * Existing non-Julia environments may be present, but they are reference
    material only during this pass.
* **Toolchain paths on this box** (Blackwell RTX 5090):
  * NVIDIA HPC SDK: `/opt/nvidia/hpc_sdk/Linux_x86_64/26.3` → `nvcc`,
    `nvc++`. Use `ARCH=sm_120` for CUDA and `SM=cc120` for OMP nvc++.
  * No local ROCm — HIP builds must be marked `skipped`.
  * No local oneAPI — SYCL builds must be marked `skipped`.
* **Benchmark metadata**: `benchmarks.yaml` at the repo root lists each
  benchmark's category, existing model set, and (for many) the default
  `args` and stdout-parsing regex used by the perf tools.

---

## 3. Scope — Julia coverage for the full CUDA catalogue

**Status: translation tranche complete/triaged.** The previous primary
objective was to create working `src/<bench>-julia/` ports for CUDA
benchmarks without Julia siblings. That objective is finished for the current
pass: the tree has 448 Julia directories, the coverage database records 441
Julia `ok` rows, five existing Julia ports still have `run_fail` rows, and
all 89 CUDA benchmarks still lacking a Julia directory have been recorded as
deferred or out of scope in local state. Future agents should not continue
walking the missing-port list by default.

Only create or repair Julia ports now when the user explicitly asks to reopen
a specific failing or deferred benchmark. Otherwise, use the worklist and
verification commands in this section only as historical context and for
targeted rechecks while the active work proceeds through the numerical
accuracy audit in §11.

Regenerate the exact missing count with the §3.1 commands — it drifts as
ports land. Outside the explicit parallel workflow in §3.2.2 and §4.0, do not
open a second port while one is still failing verification.

### 3.1 The Julia worklist

Regenerate the Julia worklist every time you sit down — some ports may have
landed since the last pass:

```bash
ls src/*-cuda   -d | xargs -n1 basename | sed 's/-cuda$//'   | sort > ${STATE}/cuda.txt
ls src/*-omp    -d | xargs -n1 basename | sed 's/-omp$//'    | sort > ${STATE}/omp.txt
ls src/*-julia  -d 2>/dev/null | xargs -n1 basename | sed 's/-julia$//'  | sort > ${STATE}/julia.txt

# Julia gap
comm -23 ${STATE}/cuda.txt ${STATE}/julia.txt  > ${STATE}/julia_missing.txt

wc -l ${STATE}/julia_missing.txt
```

Persist these lists; the `manifest.json` per port (§6) is what tells you
which entries have already been *attempted* (successfully or not) so you
don't restart from zero on the next session.

### 3.2 Prioritization inside the worklists

Do **not** walk the lists alphabetically. Rank benchmarks so you spend
LLM tokens where they pay off:

#### 3.2.1 Julia prioritization

Apply to `${STATE}/julia_missing.txt`.

1. **Batch A — trivially portable, first**. Benchmarks whose CUDA
   sources are (a) single `.cu` file, (b) < ~400 LoC, (c) no external
   libraries. Grep-friendly test:
   ```bash
   for b in $(cat ${STATE}/julia_missing.txt); do
     files=$(ls src/${b}-cuda/*.cu 2>/dev/null | wc -l)
     loc=$(wc -l src/${b}-cuda/*.cu 2>/dev/null | awk 'END{print $1}')
     ext=$(grep -lE 'boost|gsl|gdal|mpi|nccl|ccl|bz2' src/${b}-cuda/*.cu src/${b}-cuda/*.h 2>/dev/null | wc -l)
     [ "$files" = 1 ] && [ "$loc" -lt 400 ] && [ "$ext" = 0 ] && echo "$b"
   done > ${STATE}/batch_A.txt
   ```
2. **Batch B — has a self-verifier**. From what's left, keep only ports
   whose CUDA source contains `compare_results`, `verify(`, or prints
   `PASS`/`FAIL`. These give you a signal without you having to invent a
   CPU reference:
   ```bash
   for b in $(comm -23 ${STATE}/julia_missing.txt ${STATE}/batch_A.txt); do
     grep -qE 'compare_results|\bverify\b|"PASS"|"FAIL"' src/${b}-cuda/*.{cu,h,cpp} 2>/dev/null && echo "$b"
   done > ${STATE}/batch_B.txt
   ```
3. **Batch C — medium complexity**. The rest of the worklist, minus
   anything on the deferred list (§3.3).
4. **Batch D — deferred**. Do not attempt in this pass; see §3.3.

Work Batch A first, then Batch B, then Batch C. Outside the explicit
sub-agent workflow below, do not open a new port while one is still failing
verification.

#### 3.2.2 Current 20-to-50 benchmark verified short-list

For the current user-requested push, first finish the self-verifying
single-file CUDA benchmarks under 400 LoC. Regenerate this list after every
batch because completed ports disappear from `${STATE}/julia_missing.txt`:

```bash
for b in $(cat ${STATE}/julia_missing.txt); do
  files=$(ls src/${b}-cuda/*.cu 2>/dev/null | wc -l)
  loc=$(wc -l src/${b}-cuda/*.cu 2>/dev/null | awk 'END{print $1}')
  ext=$(grep -lE 'boost|gsl|gdal|mpi|nccl|ccl|bz2' \
        src/${b}-cuda/*.cu src/${b}-cuda/*.h 2>/dev/null | wc -l)
  ver=$(grep -lE 'PASS|FAIL|compare_results|\bverify\b' \
        src/${b}-cuda/*.{cu,h,cpp} 2>/dev/null | wc -l)
  [ "$files" = 1 ] && [ "$loc" -lt 400 ] && \
    [ "$ext" = 0 ] && [ "$ver" -gt 0 ] && printf '%04d %s\n' "$loc" "$b"
done | sort -n > ${STATE}/batch_verified_lt400.txt
wc -l ${STATE}/batch_verified_lt400.txt
```

Treat the regenerated list as the first candidate pool for the current
tranche. Each agent run should complete at least 50 additional Julia ports
when enough feasible CUDA benchmarks remain. If fewer than 50 candidates are
available in this verified-under-400-LoC pool, continue into Batch A, then
Batch B, then Batch C, skipping only benchmarks that meet the deferred rules
in §3.3. If regenerated counts differ from previous runs, trust the file
tree and record the new counts in `${STATE}/logs/worklist_YYYY-MM-DD.log`.

Parallel execution rules for this tranche:

* The coordinator assigns each sub-agent a small disjoint slice, preferably
  3-5 benchmarks at a time. Assign from
  `${STATE}/batch_verified_lt400.txt` first, then refill workers from
  Batch A/B/C worklists as needed to reach the 50-port objective.
* Each sub-agent owns only its assigned `src/<bench>-julia/` directories and
  matching `${STATE}/porting_logs/<bench>-julia/` paths. It must not edit
  unrelated benchmarks, shared tools, existing Julia ports, or non-Julia
  targets unless the coordinator explicitly reassigns ownership.
* Sub-agents may implement and run native `make run` checks independently.
  The coordinator performs or re-runs final
  `tools/verify_coverage.py <bench> --models cuda,julia --only-existing`
  checks before committing.
* GPU verification can be resource-bound. If parallel CUDA/JIT runs become
  unstable, serialize the final verifier commands while keeping code
  implementation parallel.
* If a sub-agent hits a failing port that needs more than two repair
  iterations, it writes an in-flight checkpoint, records the current error
  tag, and returns that benchmark to the coordinator instead of blocking the
  whole tranche.

### 3.3 Deferred / out-of-scope benchmarks

Skip and record in **`${STATE}/julia_deferred.txt`** with a one-word reason
anything that hits:

* **External native libs we don't have on this box**: Boost, GSL, GDAL,
  MPI/NCCL, CCL, BZip2. `README.md` at the repo root lists which
  benchmarks pull each of those in. The corresponding Julia bindings
  either don't exist or aren't worth the yak-shave.
* **cuBLAS / cuFFT / cuRAND / cuSPARSE / cuDNN calls in the CUDA source**
  — CUDA.jl has `CUBLAS`, `CUFFT`, `CURAND`, `CUSPARSE`, `CUDNN`
  submodules that mirror these. **Prefer them** to hand-rolled kernels
  for the Julia port. Only defer if the CUDA source uses a library API
  that CUDA.jl doesn't expose (grep the CUDA.jl source under
  `~/.julia/packages/CUDA/` if in doubt).
* **Multi-file kernel sources with heavy `__device__` inlining
  (>1500 LoC across `.cu` + `.cuh`)** — port cost dwarfs the paper's
  value from that data point. Note them and move on.
* **Benchmarks whose input data isn't checked in and isn't pullable
  through `dvc`** — you can't verify. Skip the Julia port.
* **Work that requires creating or repairing another target language** —
  defer with reason `not-julia-scope`.

The deferred lists *are* data points for the paper: they quantify which
CUDA idioms don't translate cheaply, and belong in the paper's §2
taxonomy.

### 3.4 Extending scope beyond Julia (not this pass)

Only if the user asks you to. Until then, non-Julia targets are frozen.

---

## 4. The Julia porting loop

### 4.0 Parallel coordinator mode for the 50-benchmark push

Use this mode only when the user has explicitly requested sub-agents or
parallel benchmark work. The coordinator remains responsible for global
correctness:

1. Regenerate `${STATE}/batch_verified_lt400.txt` (§3.2.2) plus Batch A/B/C
   worklists.
2. Remove entries that already have `src/<bench>-julia/` or a passing
   manifest in `${STATE}/porting_logs/<bench>-julia/manifest.json`.
3. Spawn workers with disjoint ownership, for example:
   ```
   Worker A owns: mcpr, bscan, zoom
   Worker B owns: channelShuffle, addBiasResidualLayerNorm, bilateral
   Worker C owns: lif, mrc, dp4a
   ```
   Workers must be told they are not alone in the codebase and must not
   revert or overwrite edits outside their assigned paths.
4. Each worker implements only its assigned ports, runs native Julia checks,
   writes manifests/checkpoints, and reports changed files plus verification
   evidence.
5. The coordinator reviews returned changes, reruns or performs final
   cross-model verification, commits passing ports in coherent batches, and
   updates local checkpoints with commit SHAs.
6. Failed or oversized ports are not allowed to stall the tranche. Record
   their taxonomy tag and move them to `${STATE}/julia_deferred.txt` or an
   in-flight checkpoint, then continue assigning remaining feasible entries
   until 50 more ports are complete or the feasible worklist is exhausted.

Even in parallel mode, do not accept a port as complete without native
`PASS` and cross-model `MATCH` evidence unless it is explicitly marked
deferred or in-flight.

Work **one benchmark at a time**. For this pass the only target is
`julia`. In single-agent mode, do not open a second port until the current
Julia port is verified, logged, committed, and followed by a local context
checkpoint. In parallel coordinator mode, this rule applies per worker and
per assigned benchmark directory rather than globally.
Follow the context hygiene section during the loop: store verbose evidence in
local log files and keep the active conversation short enough that no remote
compaction is needed.

### 4.1 Bootstrap the port directory

```
src/<bench>-julia/
  Makefile
  main.jl
```

* **Read the CUDA source first** (`src/<bench>-cuda/*.cu` + any `.h`) —
  that is the semantic spec.
* **Optionally skim existing siblings** (`-omp`, `-serial`, etc.) if they
  already exist; they can be useful references. Do not create or modify
  non-Julia siblings during this pass.
* **Preserve the timed region**. The CUDA source has an explicit
  `chrono::steady_clock` around `cudaDeviceSynchronize` → kernels →
  `cudaDeviceSynchronize`. The Julia port must reproduce the same timed
  region (see §4.5).
* **Preserve the verification**. If the CUDA source runs a CPU reference
  and calls `compare_results<T>` from `include/util.h`, do the same — call
  the CPU reference from the Julia port and emit a single `PASS`/`FAIL` line.
  Do not silently downgrade the check.

### 4.2 Makefile skeleton

The verify + perf tools drive **`make run`** with the CUDA benchmark's
canonical arguments. Use every run line the CUDA Makefile exercises.

```make
JULIA ?= julia
PROJECT ?= --project=/home/imo/HeCBench/src/_julia_env
LAUNCHER ?=

main:
	@true

clean:
	rm -rf __pycache__

run: main
	$(LAUNCHER) $(JULIA) $(PROJECT) main.jl <args>
```

### 4.3 Julia (`CUDA.jl`) gotchas

These aren't optional taste — they're the failure modes previous Julia ports
hit.

* **Julia (`CUDA.jl`) — READ FIRST FOR THIS PASS**
  * All indices in kernels are 1-based on the Julia side but the CUDA
    source is 0-based. Do the conversion once at the array-access site
    with `+ Int32(1)` and comment it, don't sprinkle it.
  * Kernel functions must return `nothing` (write bare `return` at the
    end).
  * Use `Int32` for anything crossing to the device. `Int` on this Linux
    box is 64-bit and blows up in NVRTC / PTX lowering. `blockIdx().x`
    et al. return `Int32` already; keep the arithmetic in `Int32`.
  * `CUDA.@allowscalar d_x[1]` is the only way to read a single device
    value on the host cheaply — use it for `over`/`done`/reduction flags.
  * For a kernel doing an atomic increment, use `CUDA.@atomic` on a
    device array element; for `atomicMax` on an `Int32` counter, prefer
    `CUDA.atomic_max!(pointer(arr, i), Int32(v))`.
  * **Match the timed region exactly.** The CUDA source does
    `cudaDeviceSynchronize(); auto t0 = steady_clock::now(); ...kernel
    launches...; cudaDeviceSynchronize(); auto t1 = ...`. Julian
    equivalent:
    ```julia
    CUDA.synchronize()
    t0 = time_ns()
    @cuda threads=... blocks=... kernel!(...)
    CUDA.synchronize()
    dt_us = (time_ns() - t0) * 1e-3
    @printf("Total kernel execution time : %f (us)\n", dt_us)
    ```
    The printed string must match the perf-sweep regex (see §4.5).
  * Prefer CUDA.jl submodules (`CUBLAS`, `CUFFT`, `CURAND`, `CUSPARSE`,
    `CUDNN`) over hand-rolling a kernel when the CUDA source is calling
    the vendor lib. That's what §3.3 refers to.
  * Struct-of-arrays vs array-of-structs: CUDA structs like
    `struct Node { int starting; int no_of_edges; };` should become two
    parallel `CuArray{Int32}` in Julia (SoA). Do **not** define a
    `struct Node` in Julia and pass a `CuArray{Node}` — CUDA.jl can't
    always lower non-primitive element types cleanly.
  * Kernel launches with `Int` fed as `N` (host `Int64`) will silently
    promote registers to 64-bit and hurt occupancy. Cast: `Int32(N)`.
  * Debugging: `CUDA.@device_code_warntype @cuda ...` shows
    type-instability that would otherwise silently produce slow code.

### 4.4 Iteration protocol (LLM-in-the-loop)

You are expected to iterate — most ports **do not compile on the first
try**. The loop is:

1. Draft the port.
2. `make -C src/<bench>-julia clean && make -C src/<bench>-julia`.
3. If build fails → classify the error (§5), fix, save the prompt and
   diff (§6), goto 2.
4. `make -C src/<bench>-julia run` with the standard input.
5. If run fails or emits `FAIL` → classify, fix, save, goto 2.
6. If run emits `PASS` → cross-check against another target with
   `tools/verify_coverage.py <bench> --models cuda,julia --only-existing`
   (§4.6).
7. If cross-check MISMATCH → classify, fix, save, goto 2.

Stop when 4–6 all pass. Record the iteration count, write the manifest,
commit the port, update the Julia worklist, and write a local context
checkpoint before starting another port in the same worker. In parallel
coordinator mode, the coordinator may batch several passing ports into one
commit if each port has independent verification evidence and a checkpoint.

### 4.5 Where the timing goes

For the perf sweep to pick up your port's number, print **one** line
whose shape is `Average [...] time [...] N.NN (us|ms|s)`. `parse_time_us`
in `perf_sweep.py` walks all matches and keeps the last one, so print it
after the last timed region, not in the middle.

### 4.6 Verification (semantic + numerical accuracy)

Two mandatory levels:

* **Bench-native** — the port itself runs a CPU reference and prints
  `PASS`/`FAIL`. Prefer element-wise `==` for integer outputs; for
  float, use the same tolerance the CUDA sibling uses (grep `compare` in
  `src/<bench>-cuda/`). If none is specified, use `abs(a-b) <=
  1e-3 * max(1, abs(b))` and note the choice in the port's docstring.
* **Cross-model** — run:
  ```
  python3 tools/verify_coverage.py <bench> --models cuda,julia --only-existing
  ```
  You may include additional already-existing models, such as `omp` or
  `serial`, as verification references when useful. Do not create missing
  non-Julia models for that purpose.
  This canonicalizes stdout: drops timing lines, keeps `PASS`/`FAIL`,
  compares numeric-stripped data lines as a multiset. A `MATCH` result
  means the ports agree structurally *and* every one of them said `PASS`.
  Coverage rows land in `${STATE}/coverage.db` — commit them to
  memory of the run, they are what feeds §7's paper tables.

---

## 5. Error taxonomy — record every hit

Every iteration that fails must be classified into exactly one bucket.
Store the record in the session scratchpad (see §6). The taxonomy is
fixed:

| Code | Bucket | Meaning |
|------|--------|---------|
| **B1** | Build: missing toolchain | Julia, CUDA, or package environment missing or wrong version. |
| **B2** | Build: syntax / type mismatch | Julia compile error or host/device type mismatch. |
| **B3** | Build: unresolved symbol / API drift | CUDA.jl function/module renamed, moved, or missing. |
| **B4** | Build: kernel compile failure | CUDA.jl/NVRTC/PTX failure inside a device kernel body. |
| **R1** | Runtime: crash / abort | Segfault, panic, `illegal memory access`, `CUDA_ERROR_*`. |
| **R2** | Runtime: silent wrong shape | Ran to completion, but printed lines differ in count/labels from CUDA. |
| **R3** | Runtime: hang / timeout | Killed by the `perf_sweep.py` per-cell timeout. |
| **N1** | Numerical: FAIL emitted | The port's own verifier said `FAIL`. |
| **N2** | Numerical: cross-model mismatch | `PASS` locally but disagrees with CUDA under `verify_coverage.py`. |
| **N3** | Numerical: tolerated drift | Elements differ but within the documented tolerance. Not a failure — recorded because it's part of the port's story. |
| **S1** | Semantic: wrong algorithm | The port implements a related but non-equivalent computation (e.g. dropped an inner term of the update rule). |
| **S2** | Semantic: wrong timed region | Compiles + verifies but the timed region is not the CUDA-equivalent one (missing sync, timer around wrong block). |
| **P1** | Portability compromise | Deliberate documented Julia deviation. Not a failure — recorded so the paper can name it. |
| **T1** | Tooling: verifier bug | The failure was in `verify_coverage.py`/perf tools, not the port. Fix the tool, re-run. |

When you fix an error, tag the commit / prompt log with the code so §7's
taxonomy table can be built by a simple `grep` later.

---

## 6. What to record — prompts, tokens, iterations, timings

For every Julia port, create one directory in the session
scratchpad:

```
${STATE}/porting_logs/<bench>-julia/
    prompt_01.md        # verbatim prompt sent to the LLM
    response_01.md      # verbatim reply (code diff or full file)
    error_01.md         # build/run/verify output that motivated the next iter
    tag_01              # taxonomy code, e.g. "B2"
    prompt_02.md
    ...
    manifest.json       # summary — see below
    context_checkpoint_01.md
```

`manifest.json` shape (write it *once* at the end of the port; update
it if you re-open the port later):

```json
{
  "bench": "bfs",
  "target": "julia",
  "driver": "codex",
  "driver_version": "codex-cli-<version>",
  "iterations": 4,
  "tokens_in": 12480,
  "tokens_out": 3910,
  "tokens_cached": 8200,
  "wall_time_s": 812,
  "error_tags": ["B2", "B4", "N2"],
  "final_status": "pass",
  "notes": "Any tolerance, vendor-library mapping, or portability caveat."
}
```

After the port is committed, write
`${STATE}/porting_logs/<bench>-julia/context_checkpoint_NN.md`. This is
the local replacement for remote/platform context compression. It should be
short but complete enough for the next Codex agent to continue without the
full conversation:

* completed benchmark and target;
* commit SHA and commit subject;
* files added or changed;
* implementation notes and any semantic deviations;
* build/run/verify commands and final results;
* manifest path and error tags;
* worklist updates;
* unrelated dirty worktree files that were intentionally left alone;
* next workflow note if the user has changed the run-book expectations.

Do not ask the user to compress context remotely, and do not wait for a
remote compact task before starting the next port once the local checkpoint is
written.
For long or troublesome ports, also write an in-flight checkpoint before the
conversation becomes large; include the active failure and the next command to
run so the task can resume from disk in a fresh session.

Token counting under Codex:

* **Codex CLI**: run with `--report-usage` (or read the `usage` block
  the CLI prints at end-of-session), and sum across iterations. Fields
  the OpenAI Responses API returns under `response.usage`:
  * `input_tokens` → `tokens_in`
  * `output_tokens` → `tokens_out`
  * `input_tokens_details.cached_tokens` → `tokens_cached`
* **Direct OpenAI Responses/Chat API**: read the same `usage` object
  from every response, sum, and record. Chat-Completions still returns
  `prompt_tokens` / `completion_tokens` — map those to the same fields.
* **Interactive session with no exposed counter**: approximate with
  `wc -c prompt_*.md response_*.md` and set
  `notes: "token count approximate (wc -c)"`.

Always set `driver: "codex"` and populate `driver_version` (the CLI
version string, or the model id if you're calling the API directly).
The paper's per-driver split in §7 depends on this field being set on
every port you produce.

Runtime performance goes into `perf.csv` (via `perf_sweep.py`), not the
per-port log. That keeps the port log about *cost of producing the
port* and the CSV about *cost of executing it*.

---

## 7. Feeding the paper

The paper skeleton is `Hecbench-agent-paper/IEEEtran/` — sections 1, 2,
and 4 are currently empty.

**Per-driver cost-study separation.** Every port produced by Codex is
recorded with `"driver": "codex"` in its `manifest.json` (§6). The
paper's cost-study tables (iterations, tokens, error taxonomy hits)
have *separate columns* for Claude and Codex — do not merge them, and
**do not overwrite a Claude-authored port's manifest** with Codex-run
metrics. If you re-verify a Claude-authored port under Codex (fine,
encouraged as a checkpoint), append a `"reverified_by": "codex"` field
plus a fresh timestamp but leave the original iteration/token counts
alone.

* **§4 performance comparison** — build with:
  ```
  python3 tools/perf_sweep.py \
    --benches accuracy adam adjacent atan2 aidw \
    --models cuda julia \
    --out ${STATE}/perf.csv
  python3 tools/perf_sweep_to_tex.py
  # → ${STATE}/perf_rows.tex, paste into 4.perfomrance_comparison.tex
  ```
  Extend `BENCHMARKS`/`MODELS` in `perf_sweep.py` when you add a new row.
  For the Julia-coverage pass specifically, you do **not** need to add
  every one of the 505 newly ported benchmarks to the perf sweep — that
  would balloon the table beyond what the paper can render. Pick a
  representative subset (roughly 20–30, covering algorithms/graph/
  linear-algebra/image/crypto categories per `benchmarks.yaml`) once
  the coverage sweep is done.
* **§2 taxonomy of porting errors** — one row per `error_tags` bucket
  from §5 for Julia ports, cell = count of ports that hit that bucket
  at least once. No script exists yet; a small Python script over
  `${STATE}/porting_logs/*-julia/manifest.json` produces it.
  Write that script the first time the paper asks for the table, name it
  `tools/taxonomy_to_tex.py`, and commit it.
* **§1/§2 Julia coverage table** — new for this pass. Two columns
  (Julia-Claude, Julia-Codex),
  rows are: "successfully ported" (target `PASS` + cross-model
  `MATCH`), "in flight" (built, not verified), and "deferred /
  out-of-scope" broken down by reason from §3.3 (`external-lib`,
  `vendor-lib-unmapped`, `oversized`, `no-input`, `not-julia-scope`).
  Source of truth is `${STATE}/coverage.db` (`verify_status` table)
  plus the deferred lists plus the `driver` field in each
  `manifest.json`. Build the table with a small Python script — commit
  it as `tools/coverage_to_tex.py` and let it take
  `--target julia` and `--driver claude|codex|all` flags.
* **§2 taxonomy per driver** — the error-taxonomy table
  (§5 buckets for Julia) is also **duplicated per driver** in the paper.
  When you extend `tools/taxonomy_to_tex.py`, take the same `--driver`
  flag as `coverage_to_tex.py`.

The **diagram** (system diagram in §1) is not yet in the tex tree. If the
user asks you to draft it, sketch it as a Mermaid or TikZ block that
mirrors §4's loop: `CUDA source → LLM iteration loop → port + logs →
verify_coverage / perf_sweep → paper tables`.

---

## 8. Commit convention

One commit per Julia port. Message form (from `git log`):

```
<bench>-julia: Julia port (CUDA.jl)

Generated-By: OpenAI Codex
```

Existing commits on this branch may carry a
`Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>`
trailer instead — that is how the paper's per-driver split in §7
resolves ties when a `manifest.json` is missing. Use the `Generated-By`
trailer above for every Codex-authored commit so the two are easy to
partition:

```bash
git log --grep='Generated-By: OpenAI Codex' --oneline   # Codex-authored
git log --grep='Co-Authored-By: Claude'   --oneline     # Claude-authored
```

Examples that already exist:

```
softmax-julia: Julia port (CUDA.jl)
```

Rules:

* Stage only the files under `src/<bench>-julia/` plus any tooling
  change the Julia port needed. Do **not** stage `build/`, `.stamps/`,
  `target/`, or `__pycache__/` — `.gitignore` already excludes them but
  a stray `git add -A` will pull them in.
* Do not commit `${STATE}/**` — the porting logs and coverage db live
  under `.porting-state/` (gitignored) by design.
* Never `--amend` a published commit. If a port has to be revised, make
  a new commit `<bench>-julia: fix <what>` on top. Never touch a
  commit produced under the other driver — the paper's cost columns
  depend on those commits being stable, and rewriting one loses the
  `Co-Authored-By` / `Generated-By` provenance.
* After committing, write
  `${STATE}/porting_logs/<bench>-julia/context_checkpoint_NN.md` before
  continuing to the next port.

---

## 9. Worked examples

### 9a. Julia — porting one missing benchmark end-to-end

Take the first entry of `${STATE}/batch_A.txt` — say it is `xsbench`
(illustrative). `src/xsbench-cuda/` exists, `src/xsbench-julia/` doesn't.

1. `ls src/xsbench-cuda/` — inspect the source layout, note `.cu`,
   headers, any `data/xsbench/*` input file, and the `run:` target in
   its Makefile (that's the canonical input args). Confirm no external
   libs are pulled in (grep for
   `boost|gsl|gdal|mpi|nccl|ccl|bz2|cublas|cufft`).
   If cuBLAS/cuFFT/cuRAND/cuSPARSE/cuDNN show up, plan to use the
   matching CUDA.jl submodule (§3.3) instead of hand-rolling.
2. Read the CUDA source fully. Identify: (a) the kernel(s), (b) the
   timed region, (c) the CPU verifier + tolerance.
3. Optionally read existing siblings for verifier and host-loop context. Do
   not create or modify non-Julia siblings.
4. Create `src/xsbench-julia/{main.jl, Makefile}` from the templates in
   §4.2. Write kernels as `@cuda` functions returning `nothing`; keep
   the CUDA kernel structure line-for-line so the two are easy to diff.
   Watch the Julia gotchas in §4.3 (1-based indexing, `Int32`, SoA over
   AoS, matching timed region).
5. `make -C src/xsbench-julia run` — expect a first-try failure. Common
   ones: `Int64`/`Int32` mismatch across the kernel boundary (`B2`),
   `CUDA.jl` API name (`B3`), off-by-one in the 1-based/0-based
   translation (`N1` or `N2`).
6. Save each `(prompt, response, error, tag)` under
   `${STATE}/porting_logs/xsbench-julia/` per §6.
7. When `make run` prints `PASS`, run:
   ```
   python3 tools/verify_coverage.py xsbench \
     --models cuda,julia --only-existing
   ```
   Expect `MATCH`. If `MISMATCH` and the mismatch is float noise, either
   (i) tighten `compare_outputs`'s canonicalizer (that's a `T1`), or
   (ii) accept it as `N3` and document in the port's docstring.
8. Write `manifest.json`, then commit:
   ```
   xsbench-julia: Julia port (CUDA.jl)
   ```
9. Update the worklist:
   ```
   grep -v '^xsbench$' ${STATE}/julia_missing.txt \
     > ${STATE}/julia_missing.txt.tmp && \
     mv ${STATE}/julia_missing.txt{.tmp,}
   ```
10. Write a local checkpoint:
    ```
    ${STATE}/porting_logs/xsbench-julia/context_checkpoint_01.md
    ```
    Include commit SHA, files changed, final verification output, manifest
    path, worklist updates, and remaining unrelated worktree changes.
11. Start the next entry of Batch A only after that local checkpoint exists.
    Do not ask the user to run a remote/platform context compression task.

Mass-cadence expectation: a Batch A benchmark should take under an hour
of wall-clock and under ~30k tokens per port when it's genuinely
trivial. If any single port blows past 2× either budget, mark it deferred
(§3.3, tag `oversized`) and move on — the paper's story is the
distribution of cost, not chasing every long-tail case.

---

## 10. Fast reference — the commands you'll use most

```bash
# 1. Regenerate the Julia worklist (do this every session)
ls src/*-cuda   -d | xargs -n1 basename | sed 's/-cuda$//'   | sort > ${STATE}/cuda.txt
ls src/*-omp    -d | xargs -n1 basename | sed 's/-omp$//'    | sort > ${STATE}/omp.txt
ls src/*-julia  -d 2>/dev/null | xargs -n1 basename | sed 's/-julia$//'  | sort > ${STATE}/julia.txt
comm -23 ${STATE}/cuda.txt ${STATE}/julia.txt  > ${STATE}/julia_missing.txt
wc -l ${STATE}/julia_missing.txt

# 2. Build Julia Batch A (single-file, <400 LoC, no external libs)
for b in $(cat ${STATE}/julia_missing.txt); do
  files=$(ls src/${b}-cuda/*.cu 2>/dev/null | wc -l)
  loc=$(wc -l src/${b}-cuda/*.cu 2>/dev/null | awk 'END{print $1}')
  ext=$(grep -lE 'boost|gsl|gdal|mpi|nccl|ccl|bz2' src/${b}-cuda/*.{cu,h} 2>/dev/null | wc -l)
  [ "$files" = 1 ] && [ "$loc" -lt 400 ] && [ "$ext" = 0 ] && echo "$b"
done > ${STATE}/batch_A.txt
wc -l ${STATE}/batch_A.txt

# 2b. Current parallel tranche: verified single-file, <400 LoC candidates
for b in $(cat ${STATE}/julia_missing.txt); do
  files=$(ls src/${b}-cuda/*.cu 2>/dev/null | wc -l)
  loc=$(wc -l src/${b}-cuda/*.cu 2>/dev/null | awk 'END{print $1}')
  ext=$(grep -lE 'boost|gsl|gdal|mpi|nccl|ccl|bz2' src/${b}-cuda/*.{cu,h} 2>/dev/null | wc -l)
  ver=$(grep -lE 'PASS|FAIL|compare_results|\bverify\b' src/${b}-cuda/*.{cu,h,cpp} 2>/dev/null | wc -l)
  [ "$files" = 1 ] && [ "$loc" -lt 400 ] && [ "$ext" = 0 ] && [ "$ver" -gt 0 ] && printf '%04d %s\n' "$loc" "$b"
done | sort -n > ${STATE}/batch_verified_lt400.txt
wc -l ${STATE}/batch_verified_lt400.txt

# 3. Build + run one Julia port
make -C src/<bench>-julia clean && make -C src/<bench>-julia
make -C src/<bench>-julia run

# 4. Cross-verify one Julia port against CUDA
python3 tools/verify_coverage.py <bench> \
  --models cuda,julia --only-existing

# 5. Bulk-verify the whole existing Julia set
for b in $(ls src/*-julia -d | xargs -n1 basename | sed 's/-julia$//'); do
  python3 tools/verify_coverage.py "$b" \
    --models cuda,julia --only-existing 2>&1 \
    | tee -a ${STATE}/logs/julia_sweep.log
done

# 6. Perf sweep for a hand-picked subset
python3 tools/perf_sweep.py --benches <bench1> <bench2> ... \
  --models cuda julia

# 7. Regenerate the LaTeX perf rows for the paper
python3 tools/perf_sweep_to_tex.py

# 8. Query Julia verification history from a prior session
sqlite3 ${STATE}/coverage.db \
  "SELECT bench, model, status, detail FROM verify_status
   WHERE model = 'julia' ORDER BY bench, model;"
```

---

## 11. Active work: numerical correctness fixes

The Julia porting push has now completed the broad coverage tranche:
`${STATE}/coverage.db` currently records **441 Julia ports with verified
`ok` results** under `tools/verify_coverage.py`. That result means the
ports build/run and the existing cross-model verifier accepts the CUDA and
Julia outputs. It does **not** guarantee strict numerical equivalence for
benchmarks that lack a native `PASS`/`FAIL` verifier, because
`verify_coverage.py` falls back to structural stdout comparison after
normalizing numeric literals.

The numerical-audit tooling is now in place; the next pass is therefore a
**numerical correctness correction pass**. Treat CUDA as the reference output.
Do not replace the existing coverage verifier; use the stronger numerical
layer alongside it and fix or document every failed/gap row.

To-do for the next agent:

1. **Classify every Julia port by verification strength.**
   Split `src/*-julia/` into: (a) ports with native `PASS`/`FAIL`, (b)
   ports that only print numeric stdout summaries, (c) ports that write
   output files/images/arrays, and (d) ports that cannot be numerically
   checked until missing inputs or external dependencies are available.
   Persist the classification under `${STATE}/numeric_accuracy/` and
   summarize counts in `${STATE}/logs/numeric_accuracy_classification.log`.

2. **Add a dedicated numeric verifier.**
   Create `tools/verify_numeric.py` to build/run CUDA and Julia, capture
   stdout/stderr, compare non-timing numeric values with tolerances, compare
   configured output artifacts, and write results to
   `${STATE}/numeric_accuracy.db`. The tool should accept at least:
   ```
   python3 tools/verify_numeric.py <bench> \
     --models cuda,julia --only-existing
   ```
   and a bulk mode or shell-friendly loop should work across all existing
   Julia ports.

3. **Extend benchmark metadata for numeric checks.**
   Add `numeric_check:` entries to `benchmarks.yaml` as needed. Supported
   metadata should cover stdout regexes, relative/absolute tolerances, and
   output files. Example shape:
   ```yaml
   minimod:
     numeric_check:
       stdout_regex:
         - 'Checksum: min_u,  max_u = ([0-9eE+.-]+), ([0-9eE+.-]+)'
       rtol: 1.0e-5
       atol: 1.0e-7

   dxtc2:
     numeric_check:
       files:
         - path: data/lena_std.dds
           kind: binary
           tolerance:
             rms: 1.0e-2
   ```
   Start with the recently added/known weak-verification ports
   (`bm3d`, `dxtc2`, `memtest`, `merkle`, `minimod`, `miniWeather`) and
   then sweep the rest.

4. **Add deterministic comparable summaries when needed.**
   If CUDA and Julia do not currently print comparable numeric results,
   add a small deterministic checksum or summary only when it can be added
   consistently to both sides or is already present in the CUDA output.
   Do not silently invent a Julia-only correctness criterion and call it
   equivalent to CUDA. Any deliberate tolerance or summary choice must be
   documented in the port log and tagged `N3` or `P1` as appropriate.

5. **Run artifact-producing benchmarks in isolated directories.**
   Some benchmarks overwrite tracked or shared output files. The numeric
   verifier must run CUDA and Julia in temporary per-model work directories
   under `/tmp` or `${STATE}/numeric_accuracy/runs/`, then compare outputs
   there. Do not allow a numeric sweep to dirty tracked source directories
   with generated images, binary outputs, `.dat` files, or solver dumps.

The first numerical sweep has been run for all 441 coverage-`ok` Julia
ports. The baseline is recorded in `JULIA_CORRECTNESS_CHECK.md`,
`${STATE}/numeric_accuracy.db`, and
`${STATE}/logs/numeric_accuracy_summary_2026-09-24.md`:

```text
pass|native_pass_trusted|251
pass|numeric_stdout|37
mismatch|numeric_stdout|76
mismatch|native_verdict|1
unverified|no_numeric_data|56
not_run|run_status|19
not_run|timeout|1
```

Current correction target:

* Investigate and fix or reclassify the `77` mismatch rows.
* Add benchmark-specific numeric evidence for the `56`
  `unverified|no_numeric_data` rows.
* Repair or retest the `20` `not_run` rows.
* Keep `JULIA_CORRECTNESS_CHECK.md` synchronized with each completed batch.

### 11.1 Plan for correcting failed numerical testing

1. **Freeze the baseline.**
   Keep `JULIA_CORRECTNESS_CHECK.md` as the top-level status. Treat
   `${STATE}/numeric_accuracy.db` as the source of truth for current
   numerical status. Export and maintain three working lists under
   `${STATE}/numeric_accuracy/reports/`: mismatches, unverified/no-data, and
   not-run.

2. **Separate false positives from real failures.**
   Inspect `mismatch|numeric_stdout` rows first. Many may be timing,
   performance numbers, nondeterministic summaries, IDs, or metadata that
   `tools/verify_numeric.py` is incorrectly treating as numerical results.
   Fix verifier filtering before touching ports. Reclassify false mismatches
   as `unverified|no_numeric_data` when no real result exists, or as
   `pass|numeric_stdout` when only harmless formatting/noise caused the
   mismatch.

3. **Add benchmark-specific `numeric_check` metadata.**
   For benchmarks with real numeric summaries, add regex/tolerance rules in
   `benchmarks.yaml`. Prioritize recently added or known weak ports:
   `bm3d`, `dxtc2`, `memtest`, `merkle`, `minimod`, and `miniWeather`, then
   work through mismatch rows alphabetically or by category.

4. **Handle artifact-producing benchmarks safely.**
   Extend `tools/verify_numeric.py` to run CUDA and Julia in isolated
   temporary directories. Add artifact comparison support: binary exact
   compare, numeric text/table compare, and image RMS/MAE comparison. Use this
   for ports that write `.png`, `.dds`, `.dat`, `.nrrd`, `.pgm`, or similar
   artifacts.

5. **Repair true Julia numerical bugs.**
   For each confirmed mismatch, read the CUDA source as the spec, compare the
   Julia implementation, and fix indexing, type width, launch geometry,
   memory layout, tolerance, RNG, or algorithmic differences. Then run native
   `make run`, `tools/verify_coverage.py`, and `tools/verify_numeric.py`.
   Write a local checkpoint and commit one benchmark fix at a time.

6. **Fix not-run rows.**
   Split not-run rows by cause: CUDA build failure, CUDA run failure, Julia
   run failure, and timeout. Do not mark numerical failure until the CUDA and
   Julia pair runs. Repair Julia run failures first; record CUDA-reference
   failures separately.

7. **Resolve unverified rows.**
   For `unverified|no_numeric_data`, add comparable checksums when CUDA
   already exposes equivalent output, add artifact metadata when files are
   generated, or leave the row explicitly documented as unverified. Do not
   claim numerical correctness without comparable evidence.

8. **Use multi-agent batches.**
   Assign workers disjoint chunks by category: verifier false-positive
   cleanup, artifact-comparison support, not-run repair, confirmed Julia bug
   fixes, and metadata/checksum additions. Each worker writes local
   checkpoints under `${STATE}/numeric_accuracy/fixes/<bench>/`. Each
   committed source fix should be one benchmark per commit and must not stage
   `.porting-state/`.

9. **Regenerate the final report.**
   After each correction batch, rerun:
   ```
   python3 tools/verify_numeric.py \
     --bench-list ${STATE}/numeric_accuracy/julia_verified_ok.txt
   python3 tools/verify_numeric.py --summary
   ```
   Update `JULIA_CORRECTNESS_CHECK.md` and
   `${STATE}/logs/numeric_accuracy_summary_YYYY-MM-DD.md`. The target is to
   maximize `pass|native_pass_trusted`, `pass|numeric_stdout`, and
   artifact-pass rows, minimize true mismatches, and explicitly document
   unavoidable unverified cases.

Record final numerical-correction status in `${STATE}/numeric_accuracy.db` and
write a concise checkpoint under
`${STATE}/logs/numeric_accuracy_summary_YYYY-MM-DD.md` with counts for:
native verifier trusted, numeric stdout matched, artifact matched,
numeric mismatch, missing inputs, external dependency, and not yet
classified.

---

## 12. What NOT to do

* Do not open ports for target languages other than Julia during this pass.
* Do not port benchmarks the user hasn't asked for — if the CUDA source
  hits any §3.3 deferral condition, add it to
  `${STATE}/julia_deferred.txt` with a one-word reason and move on.
* Do not generate Serial ports as a prerequisite for Julia. Existing
  non-Julia siblings may be read or included in verification only if they are
  already present.
* Do not "improve" the CUDA reference — even if you spot a bug, treat the
  CUDA port as the spec. File the observation in the port's docstring
  and move on.
* Do not silently downgrade a `FAIL` to a `PASS` by loosening a
  tolerance. Any tolerance change goes in the port's docstring and is
  tagged `N3` in the log.
* Do not commit `.porting-state/`, the `build/` tree, the `.stamps/`
  directories, `target/`, or `__pycache__/`.
* Do not run destructive git operations (`reset --hard`,
  `push --force`, `checkout .`) without confirming with the user
  first — even in Codex's `--dangerously-bypass-approvals-and-sandbox`
  mode. Auto-approve does not remove the need to think before
  destroying local work.
* Do not touch existing `src/*-julia/` ports that were authored by another
  driver (see §0). Editing them would corrupt the paper's per-driver cost
  split. Add new Julia ports on top; leave existing ones alone unless the user
  explicitly re-scopes them to you.
* Do not rewrite git history on this branch. The 186 commits already
  on `monil/refactor_coverage` carry the `Co-Authored-By: Claude`
  provenance that §7's per-driver tables key on — rewriting them would
  break the paper's data.
* In single-agent mode, do not open a second port while a previous port is
  still failing verification — batching drops signal about which iteration
  cost went where and pollutes the `manifest.json` records. In explicit
  parallel coordinator mode (§4.0), multiple sub-agents may work
  concurrently only on disjoint assigned benchmark directories.
* Do not continue to the next Julia port in the same worker after a
  successful commit until you have written the local context checkpoint under
  `${STATE}/porting_logs/<bench>-julia/`. In parallel mode, the coordinator
  may integrate other workers' completed ports while one worker writes its
  next checkpoint.
* Do not ask the user to run a remote/platform context compression task as
  part of the normal Julia-port cadence; local checkpoints are the handoff
  mechanism.
