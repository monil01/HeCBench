# MNIST — complete source CNN, bounded original-data study

## Scope and source fidelity

This port runs the source CNN's complete learning and testing pipeline: all **19 GPU function definitions**, two epochs, and classification. The named dataset contains the first ten training and first two test images of each digit in original file order: 100 training and 20 test images. Selection preceded all current GPU measurements. The original 60,000/10,000-image CLI, generalization, and convergence are unmeasured.

The exact declared DVC archive is 11,595,448 bytes, MD5 `53c5aacb1b0e259154ef170e15bedab8`, SHA256 `0268312488e775d8b4c671d60b8fd53f8dda842d631a5e925288ce24b26711d3`. It was already available in the local historical readiness cache; no download or operating-system/package change was made. `asset_provenance.json` records member hashes and selected original indices. The five small input assets and hashes appear in `essential_assets_for_commit.json`; do not commit raw outputs or the full archive.

`original_native/` preserves every original file. `instrumented_native/` changes allocation guards and host observation hooks only; the entire GPU-function suffix is byte-identical. `native_source_preservation_audit.json` proves the original host algorithm after removing six hooks. No native mathematical correction was required.

Important source semantics:

- Global layer constructors initialize weights **before** `main` calls `srand(123)`. `source_initial_parameters.bin` is the actual native GPU readback, independently bit-matched to the default glibc `rand(1)` sequence; every trial and Julia model uses those bytes.
- S1 is a learned, shared 4×4 stride-4 filter per channel, not max pooling.
- The top delta is the negative gradient of ten independent sigmoid binary cross-entropies. C1's averaging and S1's bias averaging are positive parameter preconditioning. The printed Euclidean error is a stopping diagnostic. A separate LongDouble chain/finite-difference witness checks six parameter groups; it does not turn the source into a squared-error optimizer.
- Source launch geometry and contiguous `N*pos/size` partitions are preserved: 64×64, except `makeError` at 10×1. Each checked trial executes 4,780 custom launches and 3,560 source-equivalent zero fills.

## Verification and results

Final receipt: `.porting-state/portable_julia/runs/20261020T015045678834Z_mnist/result.json`; configuration: `study_config_jacc.json`. GPU6, V100/sm70, CUDA 12.0, Julia 1.12.6, CUDA.jl 6.2.1, KA 0.9.45, JACC 1.4.0. All measured sources are frozen.

| Version | Mean trials 2–5 (seconds) | Complete-program / device-JIT processes | Observed translation-window tokens |
|---|---:|---:|---:|
| Native CUDA | 0.036729006 | 1 GPU process; 2 native builds | Separate readiness: 3,658,472 |
| CUDA.jl | 0.041822712 | 1 / 1 | 1,507,783 |
| KA | 0.050971447 | 1 / 1 | 1,593,744 |
| JACC | 0.069770503 | 1 / 1 | 1,999,284 |

A full pipeline warmup precedes five checked trials. Only trial 1's time is excluded. All models preserve the complete learn(2)+test work, including CPU image conversion, norm, source-style diagnostics, transfers and zero fills. Completion is charged before untimed evidence capture. Added guard allocations and explicit context fences make this an **instrumented pipeline timing**, excluded from untouched full-CLI timing claims. These measurements are not isolated GPU-kernel timings or a causal framework ranking.

Every trial retains:

- 8,981,929 full numerical values, including every meaningful activation, preactivation, derivative, weight, bias, classification and diagnostic; the first 174,943 values are exact actual input readbacks/labels;
- 420 complete forward/back stage records and physical final storage for all 28 logical allocations, with 14,336 **actual** guard bytes;
- raw bytes for unused input-layer derivative buffers, explicitly outside the initialized scientific-output contract.

CUDA.jl agrees with CUDA; KA and JACC agree with both references in all five trials under the prospectively declared `atol=5e-4, rtol=0.002`. Integral classifications are effectively exact under that bound. No thresholds were changed after observation. The independent CPU auditor checks **20,123,500** local operator/update/classification values per model using LongDouble with 64 mantissa bits and source-operation error budgets. All pass; maximum budget fractions are below 0.5. The six CPU finite-difference errors are at most 1.82e-11. All full raw-stage-to-primary joins and actual guards pass.

The measured subset still misclassifies **18/20** test images in each version. This is faithful execution of the named small training run, **not** evidence of useful digit-recognition accuracy. Both epoch stopping diagnostics stay above the source threshold, so both epochs actually execute.

## Implementation and effort qualifications

All three versions have their own 19 entries and dispatch. CUDA.jl uses `CUDA.@atomic`, KA uses `Atomix.@atomic :monotonic`, and JACC uses its exported `JACC.@atomic`. No CUDA device-intrinsic extension is needed in KA/JACC. JACC explicitly requests zero shared memory and creates fresh launch specifications. The three Julia versions share validated Julia mathematical bodies and a complete Julia host pipeline; later models are ordered translations with context reuse, not independent generations.

The first native observer build failed on C++ initializer-list type deduction; a typed-pair adapter repair preserved GPU math. Auxiliary CPU path/ordering and metadata-helper failures are retained separately as shared infrastructure effort. Each target's first full GPU candidate passed. Backend compiler-invocation counts remain unknown; they are not inferred from 19 kernel definitions or one JIT process.

Tokens are prospective, context-inclusive activity counters, including repeated/cached input and coordination. They are not billing or exclusive coding costs. Verification windows overlap translation windows and must not be added. CUDA.jl created the common harness and included native evidence review; KA included prior-model evidence-join strengthening. Both are excluded from clean target-cost calibration. JACC is eligible only in the declared ordered-reuse stratum. Annotation and native readiness have separate windows; missing historical effort remains unknown.

## Replay / CPU auditing

From the repository root, with an existing CUDA/Julia environment:

```sh
python3 tools/verify_portable_julia.py --config src/mnist-julia/study_config_native.json
python3 tools/verify_portable_julia.py --config src/mnist-julia/study_config_cuda_jl.json
python3 tools/verify_portable_julia.py --config src/mnist-julia/study_config_ka.json
python3 tools/verify_portable_julia.py --config src/mnist-julia/study_config_jacc.json
```

These recorded configs include absolute local paths and immutable reuse receipts. A relocated/fresh run must adjust source/executable/evidence paths and remove `reuse_result`/`reuse_models` to collect a fresh complete series; preserve scientific inputs, criteria and scopes. Use the central runner's GPU lock; never launch the programs outside it on the shared study host.

The four IDX files can be regenerated from the exact archive without GPU/network access:

```sh
python3 src/mnist-julia/prepare_runtime_assets.py --archive /path/to/data.zip --output-dir /tmp/mnist-case
```

The initial-parameter file is an actual-readback asset, not a generator's assumed substitute. CPU auditing needs an existing NumPy installation (recorded 2.2.6). For example:

```sh
PYTHONPATH=/tmp/stencil_ports_numpy_226 python3 src/mnist-julia/audit_mathematical_outputs.py --folder .porting-state/portable_julia/mnist_primary --model jacc --output /tmp/mnist-math.json
PYTHONPATH=/tmp/stencil_ports_numpy_226 python3 src/mnist-julia/audit_actual_inputs.py --result .porting-state/portable_julia/runs/20261020T015045678834Z_mnist/result.json --folder .porting-state/portable_julia/mnist_primary --model jacc --output /tmp/mnist-inputs.json
PYTHONPATH=/tmp/stencil_ports_numpy_226 python3 src/mnist-julia/audit_BCE_gradient_witness.py --initial src/mnist-julia/source_initial_parameters.bin --case src/mnist-julia/cases/balanced100train20test --output /tmp/mnist-bce.json
```

The audits fail closed on missing/truncated assets. Raw scientific and physical files remain local under the explicit evidence folder, outside Git/Overleaf. See `paper_observations.json` for source-backed RQ1–5 facts and limitations.
