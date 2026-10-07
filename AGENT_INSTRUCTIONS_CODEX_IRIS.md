# HeCBench IRIS Porting — Agent Instructions (Codex)

> **Which file to read.** This run-book is for OpenAI Codex agents that are
> asked to translate HeCBench CUDA benchmarks to the ORNL IRIS programming
> model. It is intentionally scoped to **IRIS C/C++ ports using data memory
> (`dmem`) only**.
>
> Do not create, repair, or extend Julia, Serial, Triton, Mojo, Rust, SYCL,
> OpenMP, or native HIP ports while following this file unless the user
> explicitly changes the scope.

This document is the run-book for the next Codex-driven agent that starts the
HeCBench IRIS porting effort. Read it end-to-end before translating a
benchmark. The current bootstrap objective is deliberately small: create and
verify **three** IRIS ports first, then use the results to harden the broader
workflow before attempting the full CUDA catalogue.

Reference material:

* IRIS documentation: https://iris-programming.readthedocs.io/en/latest/
* ORNL IRIS repository: https://github.com/ORNL/iris
* The IRIS `apps/saxpy/saxpy.cpp` example shows the preferred C++ `iris::DMem`
  pattern.
* The IRIS `apps/saxpy/kernel.cu` and `apps/saxpy/kernel.hip.cpp` examples
  show the required kernel symbol and backend file layout.

## Codex context hygiene

Remote context compaction is not part of this workflow. The run-book must be
executable from local files plus git history.

Operational rules:

* Default cadence is **one benchmark per worker at a time**: translate,
  build, run, numerical accuracy-check, write attempt records, write manifest,
  write checkpoint, commit, then move to the next benchmark.
* Multiple benchmarks may be translated concurrently only when the user
  explicitly permits parallel/sub-agent work. Parallel workers must own
  disjoint benchmarks and write sets.
* Each worker owns only its assigned `src/<bench>-iris/` directory and the
  matching `${STATE}/porting_logs/<bench>-iris/` path.
* In parallel mode, each sub-agent must complete the full loop for its assigned
  benchmark independently: source snapshot, attempt snapshots, build/run logs,
  accuracy evidence, error-corpus updates, manifest, checkpoint, and a commit.
  A coordinator may serialize final review/commits if needed, but no completed
  benchmark should remain only as an uncommitted local directory.
* Before reading or pasting large logs, write them under `${STATE}/logs/` and
  summarize only decisive lines in the chat.
* After each completed IRIS port, write a local checkpoint and commit the port
  before opening the next benchmark in that worker.
* Do not batch several benchmark ports into one commit. Use one commit per
  benchmark unless a shared tooling change must be committed separately first.
* Never stage `${STATE}/`, build trees, generated device binaries, generated
  images/data files, or profiling outputs.

## 0. Current scope

Current pass: **bootstrap the IRIS workflow with exactly three CUDA-baseline
benchmarks**.

Initial recommended benchmarks:

1. `accuracy`
2. `axpby`
3. `spaxpby`

Why these three: they are small enough to expose the IRIS host/kernel/build
workflow without immediately requiring external libraries, image artifacts, or
multi-file framework ports. If one of these is blocked by missing local inputs
or platform problems, record the blocker and replace it with the next simple,
single-node CUDA benchmark only after documenting the reason.

The full objective after the bootstrap pass is to translate every CUDA
benchmark that can be represented in IRIS `dmem` form, using the CUDA
implementation as the numerical and behavioral baseline.

## 1. Goal

For each selected `src/<bench>-cuda/`, create a working
`src/<bench>-iris/` port that:

* uses the IRIS programming model with `dmem`/`iris::DMem` only;
* launches IRIS tasks against HIP kernels on AMD GPUs with `IRIS_ARCHS=HIP`;
* preserves CUDA baseline inputs, outputs, argument semantics, and correctness
  checks;
* records compile time, runtime, and numerical accuracy against the CUDA
  baseline;
* can also carry CUDA kernel source (`kernel.cu`) for IRIS where practical,
  while AMD validation always uses the HIP backend;
* writes a local manifest, checkpoint, and git commit before the agent moves on.

CUDA is the reference. IRIS ports are not considered correct just because they
run. They must match CUDA-native PASS/FAIL verdicts or benchmark-specific
numeric/artifact checks.

## 2. State directory

Set this at the start of every session:

```bash
export STATE="${PORTING_STATE_DIR:-$PWD/.porting-state}"
mkdir -p "$STATE"/{logs,porting_logs,iris}
```

`.porting-state/` is local scratch space and must not be staged. Use it for:

* `${STATE}/iris/cuda.txt` — CUDA benchmark names.
* `${STATE}/iris/iris.txt` — existing IRIS benchmark names.
* `${STATE}/iris/iris_missing.txt` — CUDA benchmarks without an IRIS sibling.
* `${STATE}/iris/iris_deferred.txt` — blocked/out-of-scope benchmarks.
* `${STATE}/logs/<bench>-iris-*.log` — build/run/verify output.
* `${STATE}/porting_logs/<bench>-iris/manifest.json` — per-port result
  metadata.
* `${STATE}/porting_logs/<bench>-iris/context_checkpoint_NN.md` — local
  handoff written after each completed port.
* `${STATE}/porting_logs/<bench>-iris/attempts/attempt_NN/` — complete
  attempt records for wrong translations and fixes (§7 and §10.1).
* `${STATE}/iris/error_corpus.jsonl` — append-only database of LLM translation
  errors and repairs (§10.2). This is local research data; do not stage it
  unless the user explicitly asks for committed artifacts.

## 3. Local IRIS reference clone

Clone IRIS outside the HeCBench source tree when examples or headers are
needed:

```bash
git clone --depth 1 https://github.com/ORNL/iris.git /tmp/iris
```

Do not vendor `/tmp/iris` into HeCBench. Use it only as reference material.
The most relevant examples are:

```text
/tmp/iris/apps/saxpy/saxpy.cpp
/tmp/iris/apps/saxpy/kernel.cu
/tmp/iris/apps/saxpy/kernel.hip.cpp
/tmp/iris/apps/saxpy/Makefile
/tmp/iris/tests/39_dmem2dmem/cpp_host.cpp
```

The SAXPY C++ example initializes `iris::Platform`, wraps host arrays with
`iris::DMem`, launches `iris::Task::kernel`, flushes output with
`task.flush_out(mem)`, and submits the task. Follow that pattern unless a
benchmark requires a lower-level C API call.

## 4. Worklist

Regenerate the worklist every time:

```bash
ls src/*-cuda -d | xargs -n1 basename | sed 's/-cuda$//' | sort > ${STATE}/iris/cuda.txt
ls src/*-iris -d 2>/dev/null | xargs -n1 basename | sed 's/-iris$//' | sort > ${STATE}/iris/iris.txt
comm -23 ${STATE}/iris/cuda.txt ${STATE}/iris/iris.txt > ${STATE}/iris/iris_missing.txt
wc -l ${STATE}/iris/iris_missing.txt
```

For the bootstrap pass, do not walk the whole list. Start with:

```bash
printf "%s\n" accuracy axpby spaxpby > ${STATE}/iris/bootstrap3.txt
```

If a benchmark already has `src/<bench>-iris/`, inspect its manifest and
reverify it; do not overwrite it unless the user explicitly asks for a re-port.

## 5. Port layout

Each port uses this directory shape:

```text
src/<bench>-iris/
  main.cpp
  kernel.hip.cpp
  kernel.cu              # optional but preferred when the CUDA kernel maps cleanly
  Makefile
  README.md              # optional notes for non-obvious translation choices
```

Use `main.cpp` for the IRIS host program. Use `kernel.hip.cpp` for the AMD
GPU backend. Add `kernel.cu` when the CUDA kernel can be retained in IRIS form
for future NVIDIA validation. Do not put native HIP runtime allocation or
native CUDA runtime allocation in `main.cpp`; memory ownership on the host side
must go through IRIS `dmem`.

## 6. IRIS `dmem` translation rules

Required host-side pattern:

```cpp
#include <iris/iris.hpp>

int main(int argc, char** argv) {
  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  // Allocate and initialize host arrays with the same data as CUDA baseline.
  iris::DMem mem_in(input_ptr, bytes);
  iris::DMem mem_out(output_ptr, bytes);

  iris::Task task;
  void* params[] = { &mem_out, &mem_in, &n };
  int pinfo[] = { iris_w, iris_r, sizeof(n) };
  size_t gws[] = { static_cast<size_t>(n) };
  size_t lws[] = { 256 };
  task.kernel("kernel_name", 1, nullptr, gws, lws, 3, params, pinfo);
  task.flush_out(mem_out);
  task.submit(iris_gpu, nullptr, 1);

  platform.finalize();
}
```

Rules:

* Use `iris::DMem` or the C API `iris_data_mem_create`; do not use plain
  `iris::Mem` for new ports.
* Use `task.flush_out(mem)` or `iris_task_dmem_flush_out` for every output
  buffer that must be visible on the host.
* Mark parameters correctly with `iris_r`, `iris_w`, or `iris_rw`.
* Scalars use `sizeof(scalar)` in the parameter info array, not `iris_r`.
* Preserve CUDA indexing semantics. Add bounds guards in kernels when global
  work size is rounded up.
* Prefer structure-of-arrays over arrays of complex structs unless the CUDA
  source already proves the struct layout is portable and HIP compiles it.
* Preserve integer widths (`int`, `int32_t`, `size_t`, `uint64_t`) when they
  affect output.
* Do not silently change RNG seeds, default problem sizes, output filenames,
  tolerances, or iteration counts.

Required kernel pattern:

```cpp
#include <hip/hip_runtime.h>

extern "C" __global__ void kernel_name(float* out, const float* in, int n) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < static_cast<size_t>(n)) out[i] = in[i];
}
```

IRIS resolves the string passed to `task.kernel` against the backend kernel
binary. The symbol name in `kernel.hip.cpp` must match exactly.

## 7. Translation loop

Every IRIS port must be produced through an explicit iteration loop. Do not
overwrite a failed translation without preserving it first. The purpose is
both to reach correct code and to build a database of common LLM translation
errors and the fixes that resolved them.

Loop:

1. **Snapshot CUDA baseline.**
   Copy the CUDA source files that guided the translation into:

   ```text
   ${STATE}/porting_logs/<bench>-iris/source_baseline/
   ```

   Include `main.cu`, local headers, Makefiles, and any source file containing
   kernels or correctness checks. Also record the exact git commit hash.

2. **Generate attempt `NN`.**
   Before building, copy the generated IRIS files into:

   ```text
   ${STATE}/porting_logs/<bench>-iris/attempts/attempt_NN/generated/
   ```

   Include `main.cpp`, `kernel.hip.cpp`, `kernel.cu`, `Makefile`, and any
   local helper files. If the attempt is produced by editing a previous
   attempt, still snapshot the full generated source after the edit and before
   running the compiler.

3. **Build and run attempt `NN`.**
   Capture complete stdout/stderr for CUDA baseline build/run, IRIS build,
   IRIS HIP run, and numerical verification under the same attempt directory:

   ```text
   attempt_NN/logs/cuda_build.log
   attempt_NN/logs/cuda_run.log
   attempt_NN/logs/iris_build.log
   attempt_NN/logs/iris_run.log
   attempt_NN/logs/verify.log
   ```

4. **Classify the failure or pass.**
   If the attempt fails, write:

   ```text
   attempt_NN/error.md
   ```

   Include the failing command, exit code, decisive compiler/runtime/numeric
   error excerpt, taxonomy tag (§15), suspected root cause, and the planned
   fix. Keep the full logs in `attempt_NN/logs/`; `error.md` is a concise
   diagnosis.

5. **Repair and iterate.**
   Apply the smallest fix that addresses the diagnosed problem, increment
   `NN`, snapshot the next generated source, and rerun the same build/run/
   verify sequence. Continue until the port is correct, explicitly deferred,
   or blocked by environment.

6. **Finalize correct code.**
   When an attempt passes build, HIP-only run, and correctness verification,
   copy the final source into:

   ```text
   ${STATE}/porting_logs/<bench>-iris/final_correct/
   ```

   Also leave the same final code in `src/<bench>-iris/` for commit.

Do not compress, delete, or rewrite earlier failed attempts. They are the
error dataset.

## 8. Build rules

Each `src/<bench>-iris/Makefile` must support:

```bash
make -C src/<bench>-iris clean
make -C src/<bench>-iris
IRIS_ARCHS=HIP make -C src/<bench>-iris run
```

The Makefile should be configurable rather than hard-coding user-specific
paths. Recommended variables:

```make
IRIS_ROOT ?= /tmp/iris
CXX ?= g++
HIPCC ?= hipcc
CXXFLAGS ?= -O3 -std=c++17 -I$(IRIS_ROOT)/include
LDFLAGS ?= -L$(IRIS_ROOT)/build/lib -liris
RUN_ARGS ?=

all: main kernel.hip

main: main.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

kernel.hip: kernel.hip.cpp
	$(HIPCC) --genco -o $@ $<

run: all
	IRIS_ARCHS=HIP ./main $(RUN_ARGS)

clean:
	rm -f main kernel.hip kernel.ptx *.o
```

Adjust library paths to match the local IRIS build. If IRIS is not built
locally, build or install it first and record the exact command and result in
`${STATE}/logs/iris_build.log`.

Do not commit compiled artifacts: `main`, `kernel.hip`, `kernel.ptx`, object
files, shared libraries, or generated outputs.

## 9. AMD GPU execution requirement

All IRIS runtime validation for this pass must run with:

```bash
IRIS_ARCHS=HIP
```

This intentionally restricts IRIS to HIP backends and AMD GPUs. Do not accept
a CPU/OpenMP fallback as a passing IRIS result for this pass. If a run only
succeeds without `IRIS_ARCHS=HIP`, classify it as an IRIS HIP failure and
record the backend selection problem.

Suggested device sanity checks:

```bash
rocminfo | head
hipcc --version
IRIS_ARCHS=HIP make -C src/<bench>-iris run
```

If the system lacks AMD GPUs or a working HIP toolchain, do not fake the run.
Record `blocked: no-hip-runtime` or `blocked: no-amd-gpu` in the manifest.

## 10. Verification workflow

For each benchmark:

1. Build and run CUDA baseline:

   ```bash
   make -C src/<bench>-cuda clean
   /usr/bin/time -f "compile_seconds %e" make -C src/<bench>-cuda
   /usr/bin/time -f "runtime_seconds %e" make -C src/<bench>-cuda run
   ```

2. Build and run IRIS HIP:

   ```bash
   make -C src/<bench>-iris clean
   /usr/bin/time -f "compile_seconds %e" make -C src/<bench>-iris
   IRIS_ARCHS=HIP /usr/bin/time -f "runtime_seconds %e" make -C src/<bench>-iris run
   ```

3. Compare CUDA and IRIS output.

   Prefer a native `PASS`/`FAIL` verdict if the CUDA benchmark already emits
   one. Otherwise add benchmark-specific numeric summaries that both CUDA and
   IRIS print, or compare output artifacts in isolated run directories.

4. Record all commands and decisive output under:

   ```text
   ${STATE}/logs/<bench>-cuda-build.log
   ${STATE}/logs/<bench>-cuda-run.log
   ${STATE}/logs/<bench>-iris-build.log
   ${STATE}/logs/<bench>-iris-run.log
   ${STATE}/logs/<bench>-iris-verify.log
   ```

`tools/verify_coverage.py` may not know about `iris` yet. The bootstrap pass
may use shell scripts or local Python helpers, but before scaling beyond three
benchmarks add first-class `iris` model support to the verifier/perf tooling.

### 10.1 Attempt record requirements

Each attempted translation must be represented by a directory:

```text
${STATE}/porting_logs/<bench>-iris/attempts/attempt_NN/
  generated/
  logs/
  error.md              # required for failed attempts
  fix.md                # required when this attempt fixes a prior failure
  result.json
```

`result.json` shape:

```json
{
  "bench": "<bench>",
  "attempt": 1,
  "status": "build_fail|run_fail|numeric_fail|ok|deferred|blocked",
  "taxonomy": ["B2"],
  "source_snapshot": "../../source_baseline",
  "generated_snapshot": "generated",
  "failing_command": "make -C src/<bench>-iris",
  "exit_code": 2,
  "error_summary": "missing iris::Task::flush_out argument type",
  "fix_summary": "changed flush call to task.flush_out(mem_out)",
  "logs": [
    "logs/iris_build.log"
  ],
  "next_attempt": 2
}
```

For the final passing attempt, `result.json` must include:

```json
{
  "status": "ok",
  "taxonomy": [],
  "accuracy_method": "native_pass|numeric_stdout|artifact",
  "final_correct_snapshot": "../../final_correct"
}
```

### 10.2 Error corpus database

After every failed attempt and after the final fix, append one JSON line to:

```text
${STATE}/iris/error_corpus.jsonl
```

Each line should be self-contained enough for later analysis:

```json
{
  "bench": "<bench>",
  "attempt": 1,
  "driver": "codex",
  "target": "iris",
  "memory_model": "dmem",
  "backend": "HIP",
  "cuda_source_files": ["src/<bench>-cuda/main.cu"],
  "generated_files": ["src/<bench>-iris/main.cpp", "src/<bench>-iris/kernel.hip.cpp"],
  "error_stage": "compile|kernel_compile|runtime|numeric|verification",
  "taxonomy": ["B2"],
  "error_excerpt": "...",
  "root_cause": "...",
  "fix_applied": "...",
  "fixed_in_attempt": 2,
  "final_status": "fixed|deferred|blocked"
}
```

If source files are large, store paths to the snapshots rather than embedding
full source text in JSON. The attempt directories contain the source, generated
code, full error logs, and final correct code.

### 10.3 Minimum wrong-attempt evidence

If the first generated translation works on the first try, still record
`attempt_01/generated/`, `attempt_01/logs/`, `attempt_01/result.json`, and
`final_correct/`. There is no artificial need to create a wrong version, but
every real wrong version must be captured before it is fixed.

If an attempt fails in multiple ways, log the first blocking failure and note
secondary observed risks. The next attempt can expose and log the next failure.

## 11. Numerical accuracy

Accuracy standard:

* CUDA is the reference model.
* If CUDA prints `PASS`, IRIS must print `PASS` or match the same underlying
  check.
* If CUDA prints numeric summaries, IRIS must match them within documented
  `rtol`/`atol`.
* If the benchmark writes files, run CUDA and IRIS in separate temporary
  directories and compare the generated artifacts.
* If no comparable numeric evidence exists, classify the port as
  `unverified|no_numeric_data`; do not call it correct.

For each bootstrap benchmark, add or update `numeric_check:` metadata in
`benchmarks.yaml` only when it is needed and when the same rule applies to
CUDA and IRIS. Do not invent an IRIS-only correctness criterion.

## 12. Performance and timing

Record three timing dimensions separately:

* `cuda_compile_seconds`
* `iris_compile_seconds`
* `cuda_runtime_seconds`
* `iris_runtime_seconds`
* optional: kernel-only timing printed by each benchmark, if both models expose
  comparable values.

Use `/usr/bin/time` for wall-clock compile/runtime timing. Preserve existing
benchmark timing output, but do not use timing-only output as correctness data.

## 13. Manifest

After each port, write:

```text
${STATE}/porting_logs/<bench>-iris/manifest.json
```

Shape:

```json
{
  "bench": "<bench>",
  "target": "iris",
  "driver": "codex",
  "source_baseline": "src/<bench>-cuda",
  "port_dir": "src/<bench>-iris",
  "memory_model": "dmem",
  "runtime_backend": "HIP",
  "runtime_env": "IRIS_ARCHS=HIP",
  "kernel_files": ["kernel.hip.cpp", "kernel.cu"],
  "status": "ok|build_fail|run_fail|numeric_fail|unverified|deferred",
  "accuracy_method": "native_pass|numeric_stdout|artifact|none",
  "cuda_compile_seconds": null,
  "iris_compile_seconds": null,
  "cuda_runtime_seconds": null,
  "iris_runtime_seconds": null,
  "attempt_count": 0,
  "passing_attempt": null,
  "attempts_dir": "${STATE}/porting_logs/<bench>-iris/attempts",
  "source_baseline_snapshot": "${STATE}/porting_logs/<bench>-iris/source_baseline",
  "final_correct_snapshot": "${STATE}/porting_logs/<bench>-iris/final_correct",
  "error_corpus": "${STATE}/iris/error_corpus.jsonl",
  "commands": [],
  "logs": [],
  "notes": []
}
```

Keep this local under `${STATE}`. Do not stage it unless the user explicitly
asks for committed state artifacts.

## 14. Checkpoint

After each completed or deferred benchmark, write:

```text
${STATE}/porting_logs/<bench>-iris/context_checkpoint_01.md
```

Include:

* benchmark name and commit hash if committed;
* files created or changed;
* CUDA baseline command/result;
* IRIS build command/result;
* `IRIS_ARCHS=HIP` run command/result;
* numeric verification method and result;
* timing values;
* attempt count, passing attempt number, and a short list of wrong attempts;
* common error/fix observations appended to `${STATE}/iris/error_corpus.jsonl`;
* remaining risks or pitfalls;
* any unrelated dirty worktree entries noticed but not touched.

## 15. Error taxonomy

Use these tags in manifests and checkpoints:

* `B1` — missing IRIS/HIP/CUDA toolchain or wrong environment.
* `B2` — IRIS host compile error.
* `B3` — HIP kernel compile/code-object error.
* `R1` — IRIS runtime cannot find the HIP backend or kernel symbol.
* `R2` — task launch, memory, or flush failure.
* `N1` — native PASS/FAIL mismatch against CUDA.
* `N2` — numeric stdout mismatch.
* `N3` — artifact mismatch.
* `N4` — no comparable numeric evidence.
* `P1` — documented IRIS portability compromise.
* `D1` — deferred due to unsupported CUDA feature or external dependency.
* `L1` — generated code used the wrong IRIS memory model (`iris::Mem`, native
  HIP/CUDA allocation, or direct device copies instead of `dmem`).
* `L2` — generated code lost CUDA argument semantics, default problem size, or
  input initialization.
* `L3` — generated kernel symbol/parameter order does not match host launch.
* `L4` — generated bounds/grid logic differs from CUDA and causes wrong
  coverage or out-of-bounds access.
* `L5` — generated verification was too weak or compared timing instead of
  numeric output.

## 16. Bootstrap execution plan

Use this order:

1. `accuracy`
2. `axpby`
3. `spaxpby`

For each:

1. Read `src/<bench>-cuda/main.cu` and any local headers.
2. Identify inputs, outputs, kernel names, grid/block geometry, and CUDA
   correctness checks.
3. Snapshot the CUDA source under
   `${STATE}/porting_logs/<bench>-iris/source_baseline/`.
4. Create `src/<bench>-iris/`.
5. Implement attempt 1: `main.cpp` with `iris::DMem` only.
6. Implement `kernel.hip.cpp`; add `kernel.cu` if the CUDA kernel maps
   cleanly.
7. Add a Makefile with `IRIS_ARCHS=HIP` in the `run` target.
8. Snapshot attempt 1 under `attempts/attempt_01/generated/`.
9. Build and run CUDA baseline.
10. Build and run IRIS with `IRIS_ARCHS=HIP`.
11. Compare outputs numerically.
12. If build/run/verification fails, write `attempt_01/error.md`,
    append `${STATE}/iris/error_corpus.jsonl`, fix the code, and repeat as
    `attempt_02`, `attempt_03`, ... until correct or deferred.
13. After success, copy final code to `final_correct/`.
14. Write the manifest and checkpoint.
15. Commit only `src/<bench>-iris/` plus any required shared tooling or
    metadata changes.
16. Confirm `git status` shows the benchmark source changes are committed
    before starting the next benchmark. Leave unrelated dirty/generated files
    alone.

Do not start benchmark 2 in the same worker until benchmark 1 is verified,
accuracy-checked, checkpointed, and committed, or explicitly deferred with a
manifest/checkpoint. Do not start benchmark 3 in the same worker until
benchmark 2 is handled the same way.

## 17. Scaling beyond three

Only after the bootstrap pass:

* add `iris` support to `tools/verify_coverage.py` and `tools/perf_sweep.py`;
* add a dedicated IRIS numeric verifier if the existing numeric verifier is not
  model-agnostic;
* regenerate the full CUDA-vs-IRIS worklist;
* rank benchmarks by simple, single-file CUDA sources first;
* defer benchmarks that require vendor libraries, unsupported CUDA runtime
  APIs, missing input datasets, or device features that IRIS/HIP cannot expose
  cleanly.

The full catalogue workflow should still use one benchmark per commit and a
checkpoint after each completed port. It must also preserve the same attempt
history/error corpus for every benchmark, because the error database is a
primary output of the IRIS translation study.

### 17.1 Parallel/sub-agent workflow

When parallel work is allowed:

* Split the worklist into disjoint benchmark names before launching workers.
* Assign each sub-agent exclusive ownership of:
  * `src/<bench>-iris/`
  * `${STATE}/porting_logs/<bench>-iris/`
  * any benchmark-specific metadata rows it must add.
* Shared tooling files (`tools/*`, repo-level build files, `benchmarks.yaml`)
  require coordinator review to avoid conflicting edits.
* Each worker must follow the same full loop: translate, verify, accuracy
  check, attempt-log, manifest, checkpoint, and commit.
* If a worker cannot commit directly, it must stop after checkpointing and
  report the exact file list and verification status to the coordinator; the
  coordinator then commits that benchmark before assigning the worker another
  benchmark.
* Do not let multiple workers edit the same benchmark, reuse the same attempt
  directory, or append ambiguous records to the error corpus. Every
  `error_corpus.jsonl` line must include the benchmark and attempt number.

## 18. What not to do

* Do not use native HIP allocation/copy/launch as the host programming model
  and call it an IRIS port.
* Do not rely on OpenMP/CPU fallback for the pass result.
* Do not modify CUDA baseline behavior unless the same change is required to
  expose a comparable deterministic correctness summary and is documented.
* Do not commit generated device binaries or benchmark output artifacts.
* Do not skip numerical comparison because stdout "looks similar".
* Do not move to another benchmark in the same worker before the current one
  is verified, accuracy-checked, checkpointed, and committed or explicitly
  deferred.
* Do not leave completed ports uncommitted while starting additional
  translations.
* Do not overwrite a wrong generated translation without first saving it under
  `attempts/attempt_NN/generated/` and logging the failure.
* Do not only save the final code; the source baseline, generated wrong code,
  compiler/runtime/numeric error, fix description, and final correct code are
  all required.
* Do not translate broad batches until the first three IRIS ports have build,
  run, HIP-only execution, timing, numeric verification, manifests, and
  checkpoints, including complete attempt histories.
