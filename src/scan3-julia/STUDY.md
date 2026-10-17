# scan3: verified four-model study observation

RQ IDs: RQ1–RQ5. Experiment IDs: E1–E5. Evidence ID: 20261017T001548457183Z_scan3.
Configuration contract: `{"n": 4096, "repeat": 10, "input": "Float32((i17)%13+1), genuinelynonzero sharedfixture", "cases": ["nativeThrust vs custompreallocatedhierarchy", "nativeCUBtempquery/alloc/scan/free vs customactualtempalloc/pipeline/release"], "outputs": "Both4096globalexclusiveprefixarrays exactFloat64encoding", "timing": "sum2wholeoperationloopmeans withendfence, includesactualpercalltemporaryallocation insecondslot"}`.
Protocol: five_trials_last_four_v1, all five correctness checks retained, timing mean trials2–5.
Evidence: `/home/imo/HeCBench_julia/.porting-state/portable_julia/runs/20261017T001548457183Z_scan3/result.json`, config.json, timings.csv, raw logs, binary artifacts.
Ontology: ontology/benchmarks/scan3.yaml; SHA256 `3fab277d33bbe0bf23cb6453a0bdd5c140f2fc77ded70cf7dc4d521827f7733d`. Original inferred ontology labels retained; no retroactive winner-based relabeling.

## Numerical confirmation
Each variant passes independent full-element CPU oracle checks and direct binary-array comparisons against native CUDA and CUDA.jl on every one of the five trials. Acceptance tolerances and all observed absolute/relative errors are preserved in result.json. NaN/nonfinite and shape/count checks are performed by the runner. Native device kernels are included unchanged; historical CUDA.jl original_main.jl and original_Makefile are retained. Parent compatibility entrypoints execute the historical original source. Study drivers change only explicitly recorded reference inputs/measurement handling, a separate phase from original translation.

The original CUDA.jl zero-fill placeholder is preserved as historical source and separately replaced with a genuine nonzero-input global exclusive scan. Native Thrust/CUB primitives are compared with an authored three-stage framework scan: this is explicitly an algorithm adaptation, not vendor-library interoperation or vendor algorithm equivalence. All8192 outputs (two4096-element slots) match CPU, native CUDA and corrected CUDA.jl exactly across five trials. The first slot uses preallocated framework temporary buffers; the second actually allocates/releases the required two temporary buffers per internal repeat, matching the operation scope of native CUB temporary query/allocation/scan/free. Native cudaMalloc/free versus Julia pooled allocation and different temporary layouts are confounds. Native Thrust owns its internal allocation/synchronization policy. Do not attribute timing differences solely to framework abstraction. Both frameworks succeed on candidate one, reusing the separately developed scan2-style algorithm; this inherited development dependency is recorded. Selected n4096, nonzero exact-integer Float32 inputs only. Original relativeL2.001 acceptance is strengthened to exact outputs for this fixture.

## Runtime: microseconds
Each trial records the synchronized launch-loop mean per internal repeat. This includes host dispatch and excludes JIT/compilation, setup, allocations, transfers, input reset, CPU checking and artifact writing. Each trial starts with the same inputs. A device warmup precedes trial1. All timing commands are serialized on GPU6 via the shared lock. Native CUDA, CUDA.jl, KA and JACC are measured under the same named contract. Julia environments pin CUDA runtime12.0; exact package/hardware evidence lives in setup and run bundles.

| Model | Trials1–5 (us) | Mean2–5 (us) |
|---|---|---|
| cuda | 37.8593, 37.8224, 48.9583, 39.5663, 41.4914 | 41.9596 |
| cuda_jl | 36.3072, 43.8113, 33.7963, 33.2023, 32.8873 | 35.9243 |
| ka | 46.5043, 58.8935, 47.9995, 49.2574, 44.4524 | 50.1507 |
| jacc | 61.2456, 76.6786, 63.7955, 63.2296, 61.9015 | 66.4013 |

## Attempts, tokens and feature support
Each new framework has one source candidate; zero compiler-feedback code repairs. Actual runner build/JIT invocations are recorded separately from timing trials; tool/reuse/setup failures are infrastructure events, not translation candidates. A five-trial timing series is not five translation attempts. Token accounting is partial source-text characters/4, with unavailable conversation/tool/reasoning/cached usage explicitly unknown; these estimates do not establish total billed cost or an abstraction cost ranking. Shared infrastructure/reference correction/paper work are separate phases. KA-first and JACC reuse make this incremental production effort, not independent translations. No unsupported framework feature was observed for this selected configuration.

## Ontology explanation and paper follow-up
Use the existing operation/access classifications as grouping evidence, with implementation choices above recorded separately. Any runtime gap is an observation. Potential launch/configuration overhead, memory access behavior, compiler output and changed specialization are competing explanations, not demonstrated causes. No measured memory traffic, occupancy or roofline position was collected in this pass. Profiling and size scaling remain pending; this small configuration cannot support broad abstraction-selection recommendations. RQ1 is answered for this contract; RQ2 cost completeness is partial; RQ3 has matched measurements; RQ4 mechanisms and RQ5 generalization remain unresolved. Coordinator must export a portable snapshot and connect this observation to claims/figures/sections.
