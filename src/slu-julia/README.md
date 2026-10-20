# SLU: full sparse factorization and solve

The selected primary configuration executes the original NICSLU preprocessing,
MC64 ordering/scaling, symbolic fill and level scheduling, full GPU numerical LU,
and original CPU solve of `A*x=ones`. It includes the bundled Sandia/Xyce ASIC
matrix (99,340 unknowns; 954,163 original and 4,008,604 filled entries), add32,
and two positive-pivot controls. The controls exercise the optional perturbation
entry points with inactive clamps; active-clamp accuracy is not certified.

The native reference has an explicit, preserved concurrency intervention:
shared-memory publication/use/reuse barriers and single-writer pivot publication.
The same add32 racecheck reports 13 warnings originally and zero after this change.
No original wrong numerical result is inferred from the warnings. The original
GPU source is untouched and archived; conditional timings are excluded from
untouched-native comparisons. Checked host preprocessing adds fail-fast status
handling and frees temporary scalar allocations outside the measured region.

## Numerical contract

Before target translation, the complete native factors and solution passed the
predeclared independent long-double sparse-product and original-system checks.
`verification_plan.md` gives all budgets. Full actual external device inputs are
losslessly encoded in a Float64 prefix while factor storage/arithmetic is Float32.
Full original 4,113,868,080-byte ASIC workspace allocation is retained. Actual
workspace bytes are read back before/after each checked execution and compressed
losslessly; every immutable device index buffer is checked again after execution.
The raw complete factors/solutions remain in each primary artifact.

The original CPU preprocessing/symbolic/solve executes outside the numeric GPU
level-loop synchronized wallclock region. This is a hybrid pipeline with retained
NICSLU C ABI and framework-authored GPU work, not a pure Julia application or a
circuit-model validation. Five full checked trials follow an untimed full warmup;
only timings 2–5 enter the mean. Original CPU-user-time prints are auxiliary.

## Clean-checkout preparation

Run from the repository root with existing GCC, g++, nvcc, OpenMP, zlib/OpenSSL
headers and the pinned Julia environments. No system installation is performed.

```sh
python3 src/slu-julia/extract_vendor.py
python3 src/slu-julia/prepare_assets.py --output /tmp/hecbench-stencil-ports-slu-fixtures-v2
flock -x /tmp/hecbench-julia-study-gpu.lock env CUDA_VISIBLE_DEVICES=6 python3 src/slu-julia/build_program.py --mode native --output /tmp/slu-native-prepare
mkdir -p /tmp/slu-preparation-work
cd /tmp/slu-preparation-work
flock -x /tmp/hecbench-julia-study-gpu.lock env CUDA_VISIBLE_DEVICES=6 /tmp/slu-native-prepare /tmp/hecbench-stencil-ports-slu-fixtures-v2 /tmp/slu-preparation-private 1
```

The asset preparer refuses an existing destination. Preserve previous captures
and use a fresh named path for another experiment; pass `--fixtures` to the
configuration generator. `expected_assets.json` stores the frozen per-file sizes/hashes; regenerated
`asset_provenance.json` records exact original archive
members and generated fixture hashes. The original archive SHA256 is
`98aaa288fb2f3a1cb7ddd09483ec6d2fd57ec020e9806bf191ad851cbb1e8a54`.

Back at the repository root:

```sh
python3 src/slu-julia/make_config.py --models cuda --output /tmp/slu-native.json
python3 src/slu-julia/run_study.py --config /tmp/slu-native.json
```

The central verifier owns the GPU lock: do not wrap `run_study.py` in another
lock. It writes new captures under `/tmp/hecbench-stencil-ports-state` and creates
canonical state aliases with storage provenance. These local files must be
archived before machine cleanup/reboot; source Git does not preserve raw captures.
No binary fixture is required in source Git: original matrices are in the tracked
native archive and controls are regenerated exactly.

## Independent saved-state replay

```sh
g++ -std=c++17 -O3 -fopenmp src/slu-julia/replay_full_state.cpp -o /tmp/slu-replay
/tmp/slu-replay ARTIFACT_F64 /tmp/hecbench-stencil-ports-slu-fixtures-v2/cases.txt
```

The replay reads every encoded input, factor and solved value and recomputes the
full factor/system budgets without calling a GPU or the original solver. Missing,
short, lossy or trailing data fails. This proves the named backward-error contract;
it does not establish forward accuracy, conditioning or circuit physics.

## Preserved preparation failure

The first manufactured pair fixtures interleaved columns, violating the original
reader's grouping requirement. A CPU witness records `CreateMatrix=-7`, followed
by success after sorting the same entries by column. The source wrapper ignored
that original error and later crashed with an empty symbolic matrix. Earlier full
ASIC/add32 oracle logs and actual workspace captures survive, but their initial
full factors/input prefix remained in RAM and were lost. They are not reconstructed
or counted as accepted trials. Fresh preparation and per-case persistence address
this owned fixture/instrumentation error, separately from GPU translation attempts.

Final per-model acceptance, attempt counters and timings belong to frozen run
receipts and `translation_checkpoint.json`; code presence alone is not verification.

## Original checksum acceptance control

`checker_counterexample.md` gives an executed CPU-only counterexample: the
untouched checksum gate passes column-sum-preserving incorrect factors, while
the independent full-product gate rejects them. This is not an observed GPU
failure. The final numerical claim requires complete factors and the original
system residual, rather than promoting checksum agreement alone.

## Four-model measurement sequence

Generate each next config only after the previous complete receipt passes. Replace
`NATIVE_RESULT`, `CUDAJL_RESULT` and `KA_RESULT` with the actual result paths
printed by the launcher; old captures are reused only with unchanged dependencies.

```sh
python3 src/slu-julia/make_config.py --models cuda,cuda_jl --reuse-result NATIVE_RESULT --output /tmp/slu-cudajl.json
python3 src/slu-julia/run_study.py --config /tmp/slu-cudajl.json
python3 src/slu-julia/make_config.py --models cuda,cuda_jl,ka --reuse-result CUDAJL_RESULT --output /tmp/slu-ka.json
python3 src/slu-julia/run_study.py --config /tmp/slu-ka.json
python3 src/slu-julia/make_config.py --models cuda,cuda_jl,ka,jacc --reuse-result KA_RESULT --output /tmp/slu-jacc.json
python3 src/slu-julia/run_study.py --config /tmp/slu-jacc.json
```

CUDA.jl uses the KA pinned environment for CUDA6.2.1; JACC uses its separately
pinned environment. `environment.json` records exact package and file identities.
The native/CUDA.jl-only runner can return code2 because KA/JACC were not requested;
that is not a failed native check. Read each result's complete/status/trial evidence
and reference agreement explicitly. The final all-four result requires both
portable variants' dual comparisons and matching declared timing regions.

For a CPU-only twenty-artifact audit after the final receipt:

```sh
python3 src/slu-julia/audit_full_captures.py FINAL_RESULT --output /tmp/slu-all4-audit.json
```

`make -C src/slu-julia run-all FIXTURES=...` runs a fresh native→CUDA.jl→KA→JACC
campaign and checks each receipt before continuing. `make run` is a CUDA.jl-only
convenience, not four-way certification. The convenience forwarding entry point
and orchestration were added after the frozen measurement commands; those commands
execute the standard child sources directly. Reinvoking the campaign creates new
measurement series rather than overwriting or selecting earlier timings.

`actual_input_manifest.json` records the frozen full input-prefix hash, exact case
offsets and encoding. Artifact Float64 encoding does not imply Float64 GPU math.
Expected input identities are tied to the recorded compiler/NICSLU environment;
a different environment is a new named experiment, not recovery of old captures.

### Capture-path safety in new campaigns

The unmeasured configuration/orchestration helpers now choose a fresh UUID capture
root for each new campaign (or accept explicit `--capture-root`). Reused model
specifications are copied verbatim from the prior frozen config; they are not
rerun. This prevents future OS PID reuse from overwriting old private `process_PID`
paths. The measured C++ entrypoints and archived configs remain unchanged. Treat
all old capture roots as immutable and use new paths for new experiments.
