# HeCBench Multi-Language Porting — Agent Instructions (Claude Code)

> **Which file to read.** Two run-books live at the repo root:
> * `AGENT_INSTRUCTIONS_CLAUDE.md` — this file. Follow it when the
>   driver is Claude Code (or the Claude API directly).
> * `AGENT_INSTRUCTIONS_CODEX.md` — companion for OpenAI Codex.
>
> The technical guidance (§3–§5, §9, most of §10) is identical between
> the two files. The differences are the driver-specific bits: commit
> attribution, scratchpad convention, token-accounting field names, and
> the porting-log directory. Cost-study data from each driver flows
> into **separate** columns of the paper (see §7) — do not mix them.

This document is the run-book for the **next Claude-driven agent** that
continues the HeCBench cross-language porting effort. Read it
end-to-end before doing anything else — the "why" for every step is
captured, so you can make judgement calls when reality diverges from
the recipe.

**Current focus (this pass): extend two back-ends to every CUDA
benchmark in `src/*-cuda/`:**

1. **Julia (`CUDA.jl`)** — every port hand-written from the CUDA source
   (or vendor-lib call), following §4.3.
2. **Serial (single-threaded C++)** — mostly generated mechanically from
   the `-omp` sibling via `tools/omp_to_serial.py`; the remaining
   ~184 CUDA benchmarks without an OMP sibling are hand-written from
   the CUDA host loops.

The other three back-ends (Triton, Mojo, Rust) stay at the 32-benchmark
subset until this pass is complete. Read §3 before starting — most of
your work lives there.

The paper that consumes these ports is being written in
`Hecbench-agent-paper/IEEEtran/` (sections currently empty scaffolding); the
data flowing into it — verification status, execution time, iteration count,
token count, error taxonomy — is produced by the workflow described below.

---

## 1. Goal in one paragraph

HeCBench ships ~2100 kernels in CUDA / HIP / SYCL / OpenMP-target. We are
porting a **32-benchmark subset** (see §3) into **5 additional back-ends**:
Triton (Python), Julia (`CUDA.jl`), Mojo (`std.gpu.host`), Rust (`cudarc` +
NVRTC), and single-threaded CPU **Serial** (C++). For each `(benchmark,
target)` pair the agent (a) produces a working port that matches the CUDA
reference *numerically* (or by a documented ULP-tolerant rule), (b) records
the porting cost (LLM iterations, tokens, error taxonomy hits) and (c)
records runtime performance. The paper's tables and figures are built from
those records.

---

## 2. Current state — what already exists

Ground truth is the file tree, not this document; verify before trusting.

* **32 benchmarks × 5 targets are done**: list is in
  `src/*-triton/` (the Triton set is the reference for "which subset").
  Same 32 have `-julia`, `-mojo`, `-rust`, `-serial` siblings.
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
  * `omp_to_serial.py` — the OMP→Serial fallback pathway.
  * `hecbench` (CLI) — repo-wide list/build/run wrapper on the CMake tree.
* **Language environments** already set up:
  * Triton: `PYTHON=/noback/imo/miniconda3/bin/python3`, `torch` + `triton`
    installed, `torch.cuda.is_available()` is asserted in each `-triton`
    Makefile via a stamp file.
  * Julia: shared project at `src/_julia_env/` with `CUDA.jl` +
    `StaticArrays`. Every `-julia/Makefile` passes
    `--project=/home/imo/HeCBench/src/_julia_env`.
  * Mojo: pixi env at `src/_mojo_env/`, `modular >=26.4.0`. Some Mojo ports
    use a **synthetic input** instead of the CUDA benchmark's file input
    because the Mojo 1.0.0b2 string API was too rough for the file
    parsers — that's an accepted compromise (called out in the port's
    docstring).
  * Rust: cargo binary crate per bench, `cudarc = "0.13"` with
    `cuda-12080` + `nvrtc` + `driver` + `runtime` features. The CUDA
    kernel source is embedded verbatim as a `&str` and compiled by NVRTC.
  * Serial: g++ 17, `-O3`, generated from the `-omp` sibling by
    `tools/omp_to_serial.py`. `Makefile` uses `../<bench>-sycl` as an
    include path so shared `util.h`/`reference.h` still resolve.
* **Toolchain paths on this box** (Blackwell RTX 5090):
  * NVIDIA HPC SDK: `/opt/nvidia/hpc_sdk/Linux_x86_64/26.3` → `nvcc`,
    `nvc++`. Use `ARCH=sm_120` for CUDA and `SM=cc120` for OMP nvc++.
  * No local ROCm — HIP builds must be marked `skipped`.
  * No local oneAPI — SYCL builds must be marked `skipped`.
* **Benchmark metadata**: `benchmarks.yaml` at the repo root lists each
  benchmark's category, existing model set, and (for many) the default
  `args` and stdout-parsing regex used by the perf tools.

---

## 3. Scope — Julia + Serial coverage for the full CUDA catalogue

**Primary objective for this pass**: for every directory
`src/<bench>-cuda/`, create *both* a working `src/<bench>-julia/` port
and a working `src/<bench>-serial/` port. Each must (a) build cleanly,
(b) emit a bench-native `PASS`, and (c) cross-verify against the CUDA
sibling under `tools/verify_coverage.py`. Nothing else gets promoted
until this pass is done.

Sizing: `ls src/*-cuda | wc -l` currently reports **537** benchmarks.
Coverage today:

* **Julia**: 32 done, ≈**505 remaining**.
* **Serial**: 33 done, ≈**504 remaining** — of which **320** have an
  OMP sibling (mechanical `omp_to_serial.py` route) and **184** do not
  (hand-written from the CUDA host loops).

That is far too many to port in one session; the workflow below is
designed around batching and prioritization.

### 3.0 Serial-first heuristic

Do Serial before Julia for the same benchmark whenever an OMP sibling
exists. Two reasons:

* `tools/omp_to_serial.py` produces a working Serial port in seconds,
  giving you a **CPU reference** you can call from the Julia port
  without inventing one.
* The bench-native verifier the CUDA source used usually assumed a CPU
  path exists. A working `-serial` sibling is the cheapest way to keep
  that verifier honest.

When no OMP sibling exists (184 benchmarks), do Julia and Serial in
either order — the Serial port will be hand-written from the CUDA
host-code path anyway.

### 3.1 The missing-cells worklists

Regenerate both worklists every time you sit down — some ports may have
landed since the last pass:

```bash
ls src/*-cuda   -d | xargs -n1 basename | sed 's/-cuda$//'   | sort > <scratchpad>/cuda.txt
ls src/*-omp    -d | xargs -n1 basename | sed 's/-omp$//'    | sort > <scratchpad>/omp.txt
ls src/*-julia  -d 2>/dev/null | xargs -n1 basename | sed 's/-julia$//'  | sort > <scratchpad>/julia.txt
ls src/*-serial -d 2>/dev/null | xargs -n1 basename | sed 's/-serial$//' | sort > <scratchpad>/serial.txt

# Julia gap
comm -23 <scratchpad>/cuda.txt <scratchpad>/julia.txt  > <scratchpad>/julia_missing.txt
# Serial gap, split by whether OMP fallback exists
comm -23 <scratchpad>/cuda.txt <scratchpad>/serial.txt > <scratchpad>/serial_missing.txt
comm -12 <scratchpad>/serial_missing.txt <scratchpad>/omp.txt  > <scratchpad>/serial_from_omp.txt
comm -23 <scratchpad>/serial_missing.txt <scratchpad>/omp.txt  > <scratchpad>/serial_from_cuda.txt

wc -l <scratchpad>/{julia,serial}_missing.txt \
      <scratchpad>/serial_from_{omp,cuda}.txt
```

Persist these lists; the `manifest.json` per port (§6) is what tells you
which entries have already been *attempted* (successfully or not) so you
don't restart from zero on the next session.

### 3.2 Prioritization inside the worklists

Do **not** walk the lists alphabetically. Rank benchmarks so you spend
LLM tokens where they pay off:

#### 3.2.1 Serial prioritization

1. **Serial Batch M (Mechanical)** — everything on
   `<scratchpad>/serial_from_omp.txt`. Convert with:
   ```bash
   for b in $(cat <scratchpad>/serial_from_omp.txt); do
     python3 tools/omp_to_serial.py "$b" -v \
       2>&1 | tee -a <scratchpad>/logs/serial_from_omp.log
   done
   ```
   Then loop the verification: `for b in ...; do make -C
   src/${b}-serial run; done`, and for anything that fails record the
   failure taxonomy tag and fix or defer. Realistically ≥80% of these
   will build and verify on the first try — the tool is well-worn.
2. **Serial Batch H (Hand-written)** — the 184 benchmarks on
   `<scratchpad>/serial_from_cuda.txt`. Rank these using the same
   Batch A/B rules as Julia (see 3.2.2). The Serial version is written
   by stripping `__global__`/`__device__` off the CUDA kernel bodies
   and wrapping them in the equivalent for-loop over the launch grid
   dimensions — you do not need a full re-derivation from the algorithm.

#### 3.2.2 Julia prioritization

Apply to `<scratchpad>/julia_missing.txt`.

1. **Batch A — trivially portable, first**. Benchmarks whose CUDA
   sources are (a) single `.cu` file, (b) < ~400 LoC, (c) no external
   libraries. Grep-friendly test:
   ```bash
   for b in $(cat <scratchpad>/julia_missing.txt); do
     files=$(ls src/${b}-cuda/*.cu 2>/dev/null | wc -l)
     loc=$(wc -l src/${b}-cuda/*.cu 2>/dev/null | awk 'END{print $1}')
     ext=$(grep -lE 'boost|gsl|gdal|mpi|nccl' src/${b}-cuda/*.cu src/${b}-cuda/*.h 2>/dev/null | wc -l)
     [ "$files" = 1 ] && [ "$loc" -lt 400 ] && [ "$ext" = 0 ] && echo "$b"
   done > <scratchpad>/batch_A.txt
   ```
2. **Batch B — has a self-verifier**. From what's left, keep only ports
   whose CUDA source contains `compare_results`, `verify(`, or prints
   `PASS`/`FAIL`. These give you a signal without you having to invent a
   CPU reference:
   ```bash
   for b in $(comm -23 <scratchpad>/julia_missing.txt <scratchpad>/batch_A.txt); do
     grep -qE 'compare_results|\bverify\b|"PASS"|"FAIL"' src/${b}-cuda/*.{cu,h,cpp} 2>/dev/null && echo "$b"
   done > <scratchpad>/batch_B.txt
   ```
3. **Batch C — medium complexity**. The rest of the worklist, minus
   anything on the deferred list (§3.3).
4. **Batch D — deferred**. Do not attempt in this pass; see §3.3.

Order the day: Serial Batch M first (mechanical, cheap), then interleave
Serial Batch H and Julia Batch A/B/C. Do the Serial port of a given
benchmark *before* its Julia port when both are missing — Serial
becomes the CPU reference the Julia verifier calls.

Do not open a new port while one is still failing verification.

### 3.3 Deferred / out-of-scope benchmarks

Skip (and record in **`<scratchpad>/julia_deferred.txt`** and/or
**`<scratchpad>/serial_deferred.txt`** with a reason) anything that
hits:

* **External native libs we don't have on this box**: Boost, GSL, GDAL,
  MPI/NCCL, CCL, BZip2. `README.md` at the repo root lists which
  benchmarks pull each of those in. The corresponding Julia bindings
  either don't exist or aren't worth the yak-shave. Serial can still be
  attempted if the lib is CPU-side and installable (Boost, GSL) — flag
  as `needs-lib` rather than `deferred` in that case.
* **cuBLAS / cuFFT / cuRAND / cuSPARSE / cuDNN calls in the CUDA source**
  — CUDA.jl has `CUBLAS`, `CUFFT`, `CURAND`, `CUSPARSE`, `CUDNN`
  submodules that mirror these. **Prefer them** to hand-rolled kernels
  for the Julia port. Only defer if the CUDA source uses a library API
  that CUDA.jl doesn't expose (grep the CUDA.jl source under
  `~/.julia/packages/CUDA/` if in doubt). The Serial port of the same
  benchmark can call the CPU-side equivalent (`cblas_*`, FFTW, `<random>`)
  or fall back to a naive triple loop — pick whichever keeps the paper's
  cross-model numeric comparison meaningful.
* **Multi-file kernel sources with heavy `__device__` inlining
  (>1500 LoC across `.cu` + `.cuh`)** — port cost dwarfs the paper's
  value from that data point. Note them and move on. This applies to
  both targets; Serial is not automatically cheaper when the CUDA source
  is genuinely large.
* **Benchmarks whose input data isn't checked in and isn't pullable
  through `dvc`** — you can't verify. Skip both targets.
* **Serial only — benchmark's timed region is inherently multi-GPU** or
  is a communication microbenchmark (`pingpong`, `allreduce`, `ccl`,
  `halo-finder`). A single-threaded C++ port has no meaningful analogue;
  defer with reason `single-node-only`.

The deferred lists *are* data points for the paper: they quantify which
CUDA idioms don't translate cheaply, and belong in the paper's §2
taxonomy.

### 3.4 Extending scope beyond Julia + Serial (not this pass)

Only if the user asks you to. When they do, the same worklist idea
applies per target language, and the three remaining back-ends
(Triton, Mojo, Rust) have their own gotchas in §4.3 that you'll re-read
at that point.

---

## 4. The porting loop — one benchmark, one target

Work **one benchmark at a time**. For this pass the target is always
`julia` or `serial`; the framing below is generic so it stays useful
when the scope widens later. Do not open a second port until the
current one is verified and committed.

When the same benchmark needs both a Serial and a Julia port, do
**Serial first** (mechanical if `-omp` exists; hand-written from the
CUDA host loops otherwise). The Serial binary — or its CPU function —
is what the Julia port calls back to for its numerical reference,
which is much cheaper than the alternative of hand-writing a reference
inside `main.jl`.

### 4.1 Bootstrap the port directory

```
src/<bench>-<target>/
  Makefile           # `main`, `clean`, `run` targets (see §4.2)
  main.<ext>         # or main.py / main.jl / main.mojo / main.rs / <bench>.cpp
  # Rust adds: Cargo.toml, src/main.rs, target/ (build product, gitignored)
```

* **Read the CUDA source first** (`src/<bench>-cuda/*.cu` + any `.h`) —
  that is the semantic spec.
* **Also skim the OpenMP sibling** if it exists (`src/<bench>-omp/`) — its
  host loops are usually the cleanest reference for the CPU verifier and
  are what `omp_to_serial.py` operates on.
* **Preserve the timed region**. The CUDA source has an explicit
  `chrono::steady_clock` around `cudaDeviceSynchronize` → kernels →
  `cudaDeviceSynchronize`. Every port must reproduce the same timed
  region (see §4.5 for the exact rule per target).
* **Preserve the verification**. If the CUDA source runs a CPU reference
  and calls `compare_results<T>` from `include/util.h`, do the same — call
  the CPU reference from your port and emit a single `PASS`/`FAIL` line.
  Do not silently downgrade the check.

### 4.2 Makefile skeleton

The verify + perf tools drive **`make run`** with `../data/<bench>/...` as
input (or `LAUNCHER=` for numactl/nsys wrappers). Standard shapes:

* **Triton** (`Makefile`):
  ```
  PYTHON ?= /noback/imo/miniconda3/bin/python3
  LAUNCHER ?=
  $(shell mkdir -p .stamps)
  .stamps/build:
  	$(PYTHON) -c "import triton, torch; assert torch.cuda.is_available()"
  	@touch $@
  main: .stamps/build
  clean:
  	rm -rf .stamps __pycache__
  run: main
  	$(LAUNCHER) $(PYTHON) main.py <args>
  ```
* **Julia** (`Makefile`):
  ```
  JULIA ?= julia
  PROJECT ?= --project=/home/imo/HeCBench/src/_julia_env
  main: ; @true
  clean: ; @true
  run: main
  	$(LAUNCHER) $(JULIA) $(PROJECT) main.jl <args>
  ```
* **Mojo**: same shape as Julia, `mojo main.mojo <args>`.
* **Rust** (`Makefile`):
  ```
  main: ; cargo build --release
  clean: ; cargo clean
  run: main
  	$(LAUNCHER) ./target/release/<bench>-rust <args>
  ```
* **Serial** (`Makefile`): copy the `-omp` Makefile shape, drop OpenMP
  flags, keep `-I../<bench>-sycl` so shared headers resolve.

### 4.3 Language-specific gotchas (learned the hard way)

These aren't optional taste — they're the failure modes previous ports hit.
Julia and Serial are listed **first** because they are this pass's
targets; scan the others only when scope widens.

* **Serial — READ FIRST FOR THIS PASS**
  * The mechanical path is `tools/omp_to_serial.py <bench>` (or `--all`
    for the whole batch). It copies `src/<bench>-omp/` → `-serial/`,
    strips `#include <omp.h>`, drops every `#pragma omp ...` (including
    backslash-continued blocks), rewrites the Makefile as plain
    `g++ -std=c++17 -O3`, and deletes `Makefile.aomp` / `Makefile.nvc`.
    Read the tool once before trusting it — it's short.
  * **Verify every mechanical conversion**, don't just trust it. The
    common failures after `omp_to_serial.py`:
    * The `-omp` sibling was OpenMP-target-offload style (`#pragma omp
      target teams ...`), and stripping the pragmas left a bare loop
      that still assumes a device buffer view — usually shows up as a
      wrong-answer, not a build error.
    * `omp_get_wtime()` still appears after stripping (`OMP_RT_RE`
      catches this and the tool prints a warning). Replace with
      `std::chrono::steady_clock` and match the CUDA sibling's units.
    * The OMP Makefile had `-Iomp_stub` or similar; the auto-generated
      Makefile drops that but the source still `#include`s it. Point
      `-I../<bench>-sycl` at the shared headers, or add
      `-I<bench>-serial/` as needed.
  * When there is **no `-omp` sibling** you hand-write from the CUDA
    host loops. The `run_bfs_cpu` / `verify` function inside the CUDA
    file is your template — copy it verbatim, then wire up `main.cpp`
    so the same input load and result print happens as in the CUDA
    port. Do **not** invent a new algorithm.
  * Timing must be `std::chrono::steady_clock` around the same
    "expensive" region the CUDA port timed, and it must print in the
    same unit (`us`/`ms`/`s`) with the same wording so
    `perf_sweep.py`'s regex matches. Grep the CUDA sibling for
    `Average execution time` and match its format exactly.
  * Keep the OMP Makefile's `-I../<bench>-sycl` include path (the tool
    already does this) — HeCBench convention keeps shared headers like
    `util.h`, `reference.h`, `common.h` under one variant, and the
    others include across.
  * Never introduce OpenMP, TBB, or any threading library in the
    Serial port. It is single-threaded by definition — the paper uses
    it as the sequential baseline.
  * The Serial port's `PASS`/`FAIL` line **is** the reference truth for
    other back-ends' cross-model verifier. Keep it strict; don't let a
    stripped pragma silently disable a comparison.
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
* **Triton**
  * (not this pass — kept as reference for later scope extensions.)
  * `tl.static_range` is compile-time; if the CUDA loop iterates a
    data-dependent number of times, you have to bound it with a
    `MAX_DEGREE`-style clamp (see `bfs-triton/main.py`) and document it.
  * Randomness must match the CUDA seed *effect*, not the byte stream —
    C `srand(123)` and `torch.Generator.manual_seed(123)` give different
    sequences. That's fine as long as the CPU reference sees the same
    generated array.
  * Cast `int8` masks explicitly; Triton does not coerce `bool ↔ int8` the
    way CUDA does.
* **Mojo**
  * `std.gpu.host` is the current path. `DeviceContext.enqueue_function`
    launches; `map_to_host()` gives you a host-side view of a device
    buffer. No `cudaMemcpy` calls.
  * Mojo 1.0.0b2 has a fragile string API. If the benchmark reads a text
    input file and parsing is hairy, generate a **deterministic synthetic
    input** in the same shape and note this in the port's docstring — the
    verifier compares against a host BFS/reduction on the same synthetic
    input, so it's still self-consistent.
* **Rust (`cudarc`)**
  * Keep the CUDA kernel source verbatim as a `&str` and compile with
    `compile_ptx`. Do **not** rewrite the kernel in Rust — that defeats
    the point of measuring port cost.
  * Grid/block go through `LaunchConfig`; use `LaunchAsync` and a single
    `dev.synchronize()` around the timed region.
  * `char` in CUDA is `i8` in cudarc; `bool` doesn't cross.
### 4.4 Iteration protocol (LLM-in-the-loop)

You are expected to iterate — most ports **do not compile on the first
try**. The loop is:

1. Draft the port.
2. `make -C src/<bench>-<target> clean && make -C src/<bench>-<target>`
   (Rust: `cargo build --release` inside the dir).
3. If build fails → classify the error (§5), fix, save the prompt and
   diff (§6), goto 2.
4. `make -C src/<bench>-<target> run` with the standard input.
5. If run fails or emits `FAIL` → classify, fix, save, goto 2.
6. If run emits `PASS` → cross-check against another target with
   `tools/verify_coverage.py <bench> --models cuda,<target>` (§4.6).
7. If cross-check MISMATCH → classify, fix, save, goto 2.

Stop when 4–6 all pass. Record the iteration count.

### 4.5 Where the timing goes

For the perf sweep to pick up your port's number, print **one** line
whose shape is `Average [...] time [...] N.NN (us|ms|s)`. `parse_time_us`
in `perf_sweep.py` walks all matches and keeps the last one, so print it
after the last timed region, not in the middle. Serial ports must print
in `us` to match the CUDA sibling.

### 4.6 Verification (semantic + numerical accuracy)

Two mandatory levels:

* **Bench-native** — the port itself runs a CPU reference and prints
  `PASS`/`FAIL`. Prefer element-wise `==` for integer outputs; for
  float, use the same tolerance the CUDA sibling uses (grep `compare` in
  `src/<bench>-cuda/`). If none is specified, use `abs(a-b) <=
  1e-3 * max(1, abs(b))` and note the choice in the port's docstring.
* **Cross-model** — run:
  ```
  python3 tools/verify_coverage.py <bench> \
    --models cuda,omp,serial,triton,julia,rust --only-existing
  ```
  This canonicalizes stdout: drops timing lines, keeps `PASS`/`FAIL`,
  compares numeric-stripped data lines as a multiset. A `MATCH` result
  means the ports agree structurally *and* every one of them said `PASS`.
  Coverage rows land in `<scratchpad>/coverage.db` — commit them to
  memory of the run, they are what feeds §7's paper tables.

**Mojo is exempt from cross-model when it uses a synthetic input** — the
input data differs from the CUDA sibling by design. Mark those runs as
`ok` in coverage but skip them in the MATCH computation (that's what
`--only-existing` + omitting the Mojo dir buys you).

---

## 5. Error taxonomy — record every hit

Every iteration that fails must be classified into exactly one bucket.
Store the record in the session scratchpad (see §6). The taxonomy is
fixed:

| Code | Bucket | Meaning |
|------|--------|---------|
| **B1** | Build: missing toolchain | Compiler / SDK not on `PATH`, wrong version pinned in Cargo/pixi. |
| **B2** | Build: syntax / type mismatch | Language-level compile error the LLM produced. |
| **B3** | Build: unresolved symbol / API drift | Function renamed, module moved, feature flag missing (e.g. `cudarc` feature list). |
| **B4** | Build: kernel compile failure | NVRTC / Triton JIT error inside a device kernel body. |
| **R1** | Runtime: crash / abort | Segfault, panic, `illegal memory access`, `CUDA_ERROR_*`. |
| **R2** | Runtime: silent wrong shape | Ran to completion, but printed lines differ in count/labels from CUDA. |
| **R3** | Runtime: hang / timeout | Killed by the `perf_sweep.py` per-cell timeout. |
| **N1** | Numerical: FAIL emitted | The port's own verifier said `FAIL`. |
| **N2** | Numerical: cross-model mismatch | `PASS` locally but disagrees with CUDA/OMP under `verify_coverage.py`. |
| **N3** | Numerical: tolerated drift | Elements differ but within the documented tolerance. Not a failure — recorded because it's part of the port's story. |
| **S1** | Semantic: wrong algorithm | The port implements a related but non-equivalent computation (e.g. dropped an inner term of the update rule). |
| **S2** | Semantic: wrong timed region | Compiles + verifies but the timed region is not the CUDA-equivalent one (missing sync, timer around wrong block). |
| **P1** | Portability compromise | Deliberate deviation (e.g. Mojo synthetic input). Not a failure — recorded so the paper can name it. |
| **T1** | Tooling: verifier bug | The failure was in `verify_coverage.py`/perf tools, not the port. Fix the tool, re-run. |

When you fix an error, tag the commit / prompt log with the code so §7's
taxonomy table can be built by a simple `grep` later.

---

## 6. What to record — prompts, tokens, iterations, timings

For every `(bench, target)` port, create one directory in the session
scratchpad:

```
<scratchpad>/porting_logs/<bench>-<target>/
    prompt_01.md        # verbatim prompt sent to the LLM
    response_01.md      # verbatim reply (code diff or full file)
    error_01.md         # build/run/verify output that motivated the next iter
    tag_01              # taxonomy code, e.g. "B2"
    prompt_02.md
    ...
    manifest.json       # summary — see below
```

`manifest.json` shape (write it *once* at the end of the port; update
it if you re-open the port later):

```json
{
  "bench": "bfs",
  "target": "triton",
  "driver": "claude",
  "driver_version": "claude-opus-4-7",
  "iterations": 4,
  "tokens_in": 12480,
  "tokens_out": 3910,
  "tokens_cache_read": 8200,
  "tokens_cache_write": 400,
  "wall_time_s": 812,
  "error_tags": ["B2", "B4", "N2"],
  "final_status": "pass",
  "notes": "MAX_DEGREE clamp of 32 documented in main.py header."
}
```

Token counting: if you're driving via the Claude API, use the
`response.usage` fields (`input_tokens`, `output_tokens`,
`cache_read_input_tokens`, `cache_creation_input_tokens`) and sum
across iterations. If you're driving via Claude Code interactively,
approximate with `wc -c prompt_*.md response_*.md` and note the method
in `notes`. Always set `driver: "claude"` so §7's per-driver split
works.

Runtime performance goes into `perf.csv` (via `perf_sweep.py`), not the
per-port log. That keeps the port log about *cost of producing the
port* and the CSV about *cost of executing it*.

---

## 7. Feeding the paper

The paper skeleton is `Hecbench-agent-paper/IEEEtran/` — sections 1, 2,
and 4 are currently empty. Two tables are wired up:

**Per-driver cost-study separation.** Every port produced by Claude
Code is recorded with `"driver": "claude"` in its `manifest.json`
(§6). The paper's cost-study tables (iterations, tokens, error
taxonomy hits) have *separate columns* for Claude and Codex — do not
merge them, and do not overwrite a Codex-authored port's manifest with
Claude-run metrics. If you re-verify or re-run a Codex-authored port,
append a `"reverified_by": "claude"` field but leave the original
counts alone.

* **§4 performance comparison** — build with:
  ```
  python3 tools/perf_sweep.py \
    --benches accuracy adam adjacent atan2 aidw \
    --models cuda omp serial triton julia rust \
    --out <scratchpad>/perf.csv
  python3 tools/perf_sweep_to_tex.py
  # → <scratchpad>/perf_rows.tex, paste into 4.perfomrance_comparison.tex
  ```
  Extend `BENCHMARKS`/`MODELS` in `perf_sweep.py` when you add a new row.
  For the Julia-coverage pass specifically, you do **not** need to add
  every one of the 505 newly ported benchmarks to the perf sweep — that
  would balloon the table beyond what the paper can render. Pick a
  representative subset (roughly 20–30, covering algorithms/graph/
  linear-algebra/image/crypto categories per `benchmarks.yaml`) once
  the coverage sweep is done.
* **§2 taxonomy of porting errors** — one row per `error_tags` bucket
  from §5, one column per target language, cell = count of ports that
  hit that bucket at least once. No script exists yet; a five-line
  Python over `<scratchpad>/porting_logs/*/manifest.json` produces it.
  Write that script the first time the paper asks for the table, name it
  `tools/taxonomy_to_tex.py`, and commit it.
* **§1/§2 Julia + Serial coverage table** — new for this pass. Two
  columns (Julia, Serial), rows are: "successfully ported" (target
  `PASS` + cross-model `MATCH`), "in flight" (built, not verified), and
  "deferred / out-of-scope" broken down by reason from §3.3
  (`external-lib`, `vendor-lib-unmapped`, `oversized`, `no-input`,
  `single-node-only`). Source of truth is `<scratchpad>/coverage.db`
  (`verify_status` table) plus the deferred lists. Build the table
  with a small Python script — commit it as `tools/coverage_to_tex.py`
  and let it take a `--target julia|serial|all` flag so the same script
  serves both columns.
* **Serial-specific data point for §2** — split the Serial results by
  whether the port came from `omp_to_serial.py` (mechanical) or was
  hand-written from CUDA. The distribution of "first-try passes" vs
  "required iteration" between those two lanes is one of the paper's
  narratives about automation vs. LLM cost.

The **diagram** (system diagram in §1) is not yet in the tex tree. If the
user asks you to draft it, sketch it as a Mermaid or TikZ block that
mirrors §4's loop: `CUDA source → LLM iteration loop → port + logs →
verify_coverage / perf_sweep → paper tables`.

---

## 8. Commit convention

One commit per `(bench, target)` port. Message form (from `git log`):

```
<bench>-<target>: <one-line description>

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
```

Examples that already exist:

```
bfs-triton: Triton port (PyTorch + Triton)
softmax-julia: Julia port (CUDA.jl)
saxpy-ompt-omp: OpenMP port
```

Rules:

* Stage only the files under `src/<bench>-<target>/` plus any tooling
  change the port needed. Do **not** stage `build/`, `.stamps/`,
  `target/`, or `__pycache__/` — `.gitignore` already excludes them but
  a stray `git add -A` will pull them in.
* Do not commit `<scratchpad>/**` — the porting logs and coverage db live
  in `/tmp/claude-<pid>/...` and are session-scoped by design.
* Never `--amend` a published commit. If a port has to be revised, make
  a new commit `<bench>-<target>: fix <what>` on top.

---

## 9. Worked examples

### 9a. Serial from `-omp` (mechanical route)

Take the first entry of `<scratchpad>/serial_from_omp.txt` — say
`hotspot3D` (illustrative). `src/hotspot3D-cuda/` and
`src/hotspot3D-omp/` exist, `src/hotspot3D-serial/` doesn't.

1. `python3 tools/omp_to_serial.py hotspot3D -v` — creates
   `src/hotspot3D-serial/` from the OMP source. Read the tool's stderr
   for any warnings about surviving `omp_get_wtime` / continuation-line
   pragmas.
2. `make -C src/hotspot3D-serial clean && make -C src/hotspot3D-serial`
   — expect it to build clean; if not, classify (`B2`/`B3`) and fix.
3. `make -C src/hotspot3D-serial run` — expect `PASS`. If it prints a
   numeric result but no `PASS`/`FAIL`, that's `S1` in most cases (the
   OMP sibling had target-offload semantics that got mangled by pragma
   stripping) — either fix in place or defer.
4. Cross-verify:
   ```
   python3 tools/verify_coverage.py hotspot3D \
     --models cuda,omp,serial --only-existing
   ```
5. Write `manifest.json` — for mechanical Serial ports, expect
   `iterations: 1`, `error_tags: []`. That's the signal for the paper.
6. Commit:
   ```
   hotspot3D-serial: Serial port (from -omp)
   ```

### 9b. Serial hand-written from CUDA (no OMP sibling)

Same benchmark, but assume no `-omp` exists.

1. `mkdir src/hotspot3D-serial && cp src/hotspot3D-cuda/*.h
   src/hotspot3D-serial/` — start from the CUDA headers.
2. Copy the CUDA host loops (the CPU verifier + main function) into a
   new `hotspot3D.cpp`. Delete every `__global__`/`__device__` from the
   kernel; turn the CUDA kernel body into a plain `for` over the launch
   grid dimensions (product of `gridDim` × `blockDim`).
3. Replace CUDA memory management with host `malloc`/`new` (or drop
   entirely — you're operating on host arrays now).
4. Write a plain `Makefile` mirroring the shape produced by
   `omp_to_serial.py` (see §4.2 Serial template).
5. Then the same build/run/verify/commit loop as 9a. Expect more
   iterations than the mechanical case — realistic bucket is 2–3
   iterations, dominated by `B2` / `B3` fixes.

### 9c. Julia — porting one missing Julia benchmark end-to-end

Take the first entry of `<scratchpad>/batch_A.txt` — say it is `xsbench`
(illustrative). `src/xsbench-cuda/` exists, `src/xsbench-julia/` doesn't.

1. `ls src/xsbench-cuda/` — inspect the source layout, note `.cu`,
   headers, any `data/xsbench/*` input file, and the `run:` target in
   its Makefile (that's the canonical input args). Confirm no external
   libs are pulled in (grep for `boost|gsl|gdal|mpi|nccl|cublas|cufft`).
   If cuBLAS/cuFFT/cuRAND/cuSPARSE/cuDNN show up, plan to use the
   matching CUDA.jl submodule (§3.3) instead of hand-rolling.
2. Read the CUDA source fully. Identify: (a) the kernel(s), (b) the
   timed region, (c) the CPU verifier + tolerance.
3. Read `src/xsbench-omp/` — its host loops usually translate to Julia
   more cleanly than the CUDA host code.
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
   `<scratchpad>/porting_logs/xsbench-julia/` per §6.
7. When `make run` prints `PASS`, run:
   ```
   python3 tools/verify_coverage.py xsbench \
     --models cuda,omp,serial,julia --only-existing
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
   grep -v '^xsbench$' <scratchpad>/julia_missing.txt \
     > <scratchpad>/julia_missing.txt.tmp && \
     mv <scratchpad>/julia_missing.txt{.tmp,}
   ```
10. Move on to the next entry of Batch A.

Mass-cadence expectation: a Batch A benchmark should take under an hour
of wall-clock and under ~30k tokens per port when it's genuinely
trivial. If any single port blows past 2× either budget, mark it deferred
(§3.3, tag `oversized`) and move on — the paper's story is the
distribution of cost, not chasing every long-tail case.

---

## 10. Fast reference — the commands you'll use most

```bash
# 1. Regenerate the worklists (do this every session)
ls src/*-cuda   -d | xargs -n1 basename | sed 's/-cuda$//'   | sort > <scratchpad>/cuda.txt
ls src/*-omp    -d | xargs -n1 basename | sed 's/-omp$//'    | sort > <scratchpad>/omp.txt
ls src/*-julia  -d 2>/dev/null | xargs -n1 basename | sed 's/-julia$//'  | sort > <scratchpad>/julia.txt
ls src/*-serial -d 2>/dev/null | xargs -n1 basename | sed 's/-serial$//' | sort > <scratchpad>/serial.txt
comm -23 <scratchpad>/cuda.txt <scratchpad>/julia.txt  > <scratchpad>/julia_missing.txt
comm -23 <scratchpad>/cuda.txt <scratchpad>/serial.txt > <scratchpad>/serial_missing.txt
comm -12 <scratchpad>/serial_missing.txt <scratchpad>/omp.txt > <scratchpad>/serial_from_omp.txt
comm -23 <scratchpad>/serial_missing.txt <scratchpad>/omp.txt > <scratchpad>/serial_from_cuda.txt
wc -l <scratchpad>/{julia,serial}_missing.txt \
      <scratchpad>/serial_from_{omp,cuda}.txt

# 2. Serial Batch M — mechanical conversion from -omp for everything eligible
for b in $(cat <scratchpad>/serial_from_omp.txt); do
  python3 tools/omp_to_serial.py "$b" -v \
    2>&1 | tee -a <scratchpad>/logs/serial_from_omp.log
done

# 3. Build Julia Batch A (single-file, <400 LoC, no external libs)
for b in $(cat <scratchpad>/julia_missing.txt); do
  files=$(ls src/${b}-cuda/*.cu 2>/dev/null | wc -l)
  loc=$(wc -l src/${b}-cuda/*.cu 2>/dev/null | awk 'END{print $1}')
  ext=$(grep -lE 'boost|gsl|gdal|mpi|nccl' src/${b}-cuda/*.{cu,h} 2>/dev/null | wc -l)
  [ "$files" = 1 ] && [ "$loc" -lt 400 ] && [ "$ext" = 0 ] && echo "$b"
done > <scratchpad>/batch_A.txt
wc -l <scratchpad>/batch_A.txt

# 4. Build + run one port (works for both julia and serial)
make -C src/<bench>-<target> clean && make -C src/<bench>-<target> && \
make -C src/<bench>-<target> run

# 5. Cross-verify one benchmark across every model that exists locally
python3 tools/verify_coverage.py <bench> \
  --models cuda,omp,serial,julia --only-existing

# 6. Bulk-verify the whole existing Julia set (or Serial set)
for b in $(ls src/*-julia -d | xargs -n1 basename | sed 's/-julia$//'); do
  python3 tools/verify_coverage.py "$b" \
    --models cuda,omp,serial,julia --only-existing 2>&1 \
    | tee -a <scratchpad>/logs/julia_sweep.log
done
# ...and the Serial set (same shape, s/julia/serial/)

# 7. Perf sweep for a hand-picked subset (representative, not all 537)
python3 tools/perf_sweep.py --benches <bench1> <bench2> ... \
  --models cuda omp serial julia

# 8. Regenerate the LaTeX perf rows for the paper
python3 tools/perf_sweep_to_tex.py

# 9. Query verification history from a prior session (filter per target)
sqlite3 <scratchpad>/coverage.db \
  "SELECT bench, model, status, detail FROM verify_status
   WHERE model IN ('julia','serial') ORDER BY bench, model;"
```

---

## 11. What NOT to do

* Do not open ports for target languages other than Julia or Serial
  during this pass. The three remaining back-ends (Triton, Mojo, Rust)
  stay at the 32-benchmark subset until the Julia + Serial sweep is done.
* Do not port benchmarks the user hasn't asked for — if the CUDA source
  hits any §3.3 deferral condition, add it to
  `<scratchpad>/julia_deferred.txt` (and/or `serial_deferred.txt`) with
  a one-word reason and move on.
* Do not multi-thread the Serial port. `omp_to_serial.py` strips
  OpenMP; do not add TBB, `std::thread`, `std::execution::par`, or any
  other parallelism. Serial is the sequential baseline the paper
  measures against.
* Do not "improve" the CUDA reference — even if you spot a bug, treat the
  CUDA port as the spec. File the observation in the port's docstring
  and move on.
* Do not rewrite the Rust `KERNEL_SRC` in Rust. It's meant to be the
  same CUDA source, compiled by NVRTC through cudarc. (Relevant only
  when scope widens back to Rust.)
* Do not silently downgrade a `FAIL` to a `PASS` by loosening a
  tolerance. Any tolerance change goes in the port's docstring and is
  tagged `N3` in the log.
* Do not commit the scratchpad, the `build/` tree, the `.stamps/`
  directories, `target/`, or `__pycache__/`.
* Do not run destructive git operations (`reset --hard`,
  `push --force`, `checkout .`) without confirming with the user first.
* Do not open a second port while a previous port is still failing
  verification — batching drops signal about which iteration cost went
  where and pollutes the `manifest.json` records.
