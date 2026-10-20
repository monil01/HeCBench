# columnarSolver: full fixed academic fixture

All four models execute the supplied 397-character fixture, KEY_LENGTH=30,
150000 complete proposals per each of 1024 trajectories, including setupKernel.
This is conditional corrected-reference evidence, not an untouched-native or
external-input certificate. Original CUDA files remain unchanged.

## Evidence and scope

`protocol_final.json` records the predeclared exact contract and reference repairs.
`full_saved_audit.json` recomputes all 9,513,940 encoded values across 20 checked
captures, 51,200 physical guard bytes and 62 execution dependency bindings.
All raw candidate/key/score/RNG states are exact against a full CPU source replay.
Independent ragged-column reconstruction, key permutations, Float64 bigram bounds,
LCG skip-ahead and known plaintext (thread 361) also pass. Artifact Float64 is a
lossless serialization format; GPU score arithmetic is Float32.

Two original host/index hazards are established by CPU ASan and full source-bounds
replay. Owned copies use char[3]/%2s and clamp only five rounded key indices.
The intermediate instrumented native body produced 489 wrong character-zero values
on both nvcc 12.0 and 12.9 while accepted keys, scores and RNG were exact. Making
only private best plaintext storage volatile restores every byte. The compiler
root cause remains unresolved; no original unsafe GPU failure was observed.
The source winner-only checker would miss those 489 errors (CPU counterfactual,
not an actual unsafe CLI result). All failures and the independent peer review
are retained under `native_diagnostics/`.

Five checked trials follow a complete workload warmup. Means use trials 2–5.
Native / CUDA.jl / KA / JACC: 14.575939051 / 38.39174022825 /
38.3768006555 / 39.22576446775 seconds. ROI includes setup, full decode and
whole-device completion; transfers, resets, guards, readbacks and CPU checks are
outside all four. Conditional volatile storage and portable private arrays are
mapping covariates; exclude untouched-native causal performance aggregates.

## Clean-checkout regeneration

Run from repository root. Original `src/columnarSolver-cuda/data` is the 11416-byte
source score table, SHA256
`0f82eade75ec3aa52a6f31a6bc025235022dc6a1c7eab019c600647fb8fc2905`.
The exact 6100-byte original DVC archive is public at
https://hecbench-geekdude.s3.amazonaws.com/files/md5/bb/cba6be85e7d0d1cf26a58d76fea8d0
(MD5 `bbcba6be85e7d0d1cf26a58d76fea8d0`; SHA256
`d2f9ff61f9d5426486c16548485ec716ff9eb264fbefeb0bf2fb47046ad39ce4`).
Extract only its `data` member and verify the table hash. No custom input is needed.

The retained generated source is already supplied; do not rerun `prepare_study.py`
over a frozen measured checkout because it regenerates the earlier nonvolatile
adapter. Regenerate the CPU oracle from the supplied source as follows:

```bash
mkdir -p /tmp/columnarSolver_native
g++ -O3 -std=c++17 -ffp-contract=off -fopenmp \
  -DDATA_PATH=\"$(pwd)/src/columnarSolver-cuda/data\" \
  src/columnarSolver-julia/cpu_source_replay.cpp \
  -o /tmp/columnarSolver_native/cpu_separate
/tmp/columnarSolver_native/cpu_separate /tmp/columnarSolver_native/cpu_separate.bin
sha256sum /tmp/columnarSolver_native/cpu_separate.bin
```

Expected 1,888,256-byte oracle SHA256:
`8c21d8bdf4bc5773a25bd9dd5d9bfa642688fa828d621474183fbaafaddb2900`.
Compile the accepted native adapter using the existing CUDA12.0 compiler:

```bash
flock -x /tmp/hecbench-julia-study-gpu.lock env CUDA_VISIBLE_DEVICES=6 \
  /usr/local/cuda/bin/nvcc -O3 -std=c++17 -arch=sm_70 \
  -DDATA_PATH=\"$(pwd)/src/columnarSolver-cuda/data\" \
  src/columnarSolver-julia/study_native_storage_v2.cu \
  -o /tmp/columnarSolver_native/study_native_storage_v2
```

Configs preserve the original absolute checkout paths. For a relocated checkout,
make new named config copies replacing `/home/imo/HeCBench_julia` with its actual
root, preserving all explicit dependencies and contracts. Run native first,
then CUDA.jl, KA and JACC via `run_study.py --config CONFIG` (inspect its CLI).
The central verifier owns the GPU lock; never wrap it in an outer flock.
Pinned Projects/Manifests/preferences and version hashes are in
`environment_snapshot.json`. Large binaries and raw captures remain local,
regenerable `/tmp` evidence; archive them before reboot.

Saved CPU audit (requires NumPy, available locally in the listed PYTHONPATH):

```bash
PYTHONPATH=/tmp/pasc-julia-plot-tools OPENBLAS_NUM_THREADS=1 \
 python3 src/columnarSolver-julia/audit_saved.py \
 --result .porting-state/portable_julia/runs/20261020T163059377094Z_columnarSolver/result.json \
 --output /tmp/columnar_saved_recheck.json
```

## Accounting and integration

`effort_summary.json` preserves actual context-inclusive phase counters and peer
review separately. CUDA.jl creates shared Julia helpers; KA and JACC reuse them in
order. Historical 186 character-derived proxy tokens remain separate and are not
consumption. Each target has one evaluated program candidate with two GPU entry
definitions, one warmup and five checked trials; backend compilation counts are
unknown. Native adapter build failures are not target translation failures.

Automatic approval review rejected worker writes to shared canonical metadata.
`metadata_proposals/review.json` therefore contains ONLY source-owned proposals,
expected old hashes and root-review destinations. Byte-identical old records are
in `reference_history/`. Root alone reviews/applies shared metadata and commits.
No accepted sources, configs, results or measured timing were changed for metadata.
