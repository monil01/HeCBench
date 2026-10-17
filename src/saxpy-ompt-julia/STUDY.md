# saxpy-ompt: verified four-model study observation

RQ IDs: RQ1–RQ5. Experiment IDs: E1–E5. Evidence ID: 20261016T224222218766Z_saxpy-ompt.
Configuration contract: `{"precision": "float32", "fixture": "native uint32 integer initialization with zero-based index; y reset each trial", "size": 4096}`.
Protocol: five_trials_last_four_v1, all five correctness checks retained, timing mean trials2–5.
Evidence: `/home/imo/HeCBench_julia/.porting-state/portable_julia/runs/20261016T224222218766Z_saxpy-ompt/result.json`, config.json, timings.csv, raw logs, binary artifacts.
Ontology: ontology/benchmarks/saxpy-ompt.yaml; SHA256 `1d47712e62df9878e5efe90e8f48057deaa7986a04989235b4bf4596b9d12ae9`. Original inferred ontology labels retained; no retroactive winner-based relabeling.

## Numerical confirmation
Each variant passes independent full-element CPU oracle checks and direct binary-array comparisons against native CUDA and CUDA.jl on every one of the five trials. Acceptance tolerances and all observed absolute/relative errors are preserved in result.json. NaN/nonfinite and shape/count checks are performed by the runner. Native device kernels are included unchanged; historical CUDA.jl original_main.jl and original_Makefile are retained. Parent compatibility entrypoints execute the historical original source. Study drivers record reference input alignment, any algorithm correction, and measurement handling as separate phases from the historical translation.

Historical CUDA.jl initialization used one-based integer index while native used zero-based index; this study fixes the input contract in the instrumented driver. The historical source is preserved. All4096 output values agree exactly among the four GPU models; native oracle tolerance1e-2 and stricter cross-model tolerance1e-4 are distinct checks. n4096 and100 internal repeats are explicit small study inputs. JACC uses an asynchronous LaunchSpec with256threads, zero dynamicshared storage; default synchronous JACC behavior is avoided to match native loop synchronization.

## Runtime: microseconds
Five in-process trials are retained; the reported mean excludes trial1 and averages trials2–5. Each value uses the configuration's exact timed region and normalization. Notes above and `input_contract` distinguish per-repeat means from case or pipeline totals. Transfers, resets, allocation, temporary storage, graph control, host dispatch and synchronization are included where that named contract specifies. Compiler/JIT warmup, CPU correctness and artifact writing occur outside measurement. All GPU measurements use the shared GPU6 lock. Exact package/hardware evidence lives in setup and run bundles.

| Model | Trials1–5 (us) | Mean2–5 (us) |
|---|---|---|
| cuda | 2.3879, 2.3731, 2.3634, 2.3646, 2.3624 | 2.3659 |
| cuda_jl | 2.6633, 2.9029, 2.9237, 2.6172, 2.6004 | 2.7611 |
| ka | 4.8168, 4.8086, 4.5051, 4.4333, 4.1988 | 4.4865 |
| jacc | 8.1251, 8.8525, 8.1279, 8.2865, 7.9575 | 8.3061 |

## Attempts, tokens and feature support
Each new framework has one source candidate; zero compiler-feedback code repairs. Actual runner build/JIT invocations are recorded separately from timing trials; tool/reuse/setup failures are infrastructure events, not translation candidates. A five-trial timing series is not five translation attempts. Token accounting is partial source-text characters/4, with unavailable conversation/tool/reasoning/cached usage explicitly unknown; these estimates do not establish total billed cost or an abstraction cost ranking. Shared infrastructure/reference correction/paper work are separate phases. KA-first and JACC reuse make this incremental production effort, not independent translations. No unsupported framework feature was observed for this selected configuration.

## Ontology explanation and paper follow-up
Use the existing operation/access classifications as grouping evidence, with implementation choices above recorded separately. Any runtime gap is an observation. Potential launch/configuration overhead, memory access behavior, compiler output and changed specialization are competing explanations, not demonstrated causes. No measured memory traffic, occupancy or roofline position was collected in this pass. Profiling and size scaling remain pending; this small configuration cannot support broad abstraction-selection recommendations. RQ1 is answered for this contract; RQ2 cost completeness is partial; RQ3 has matched measurements; RQ4 mechanisms and RQ5 generalization remain unresolved. Coordinator must export a portable snapshot and connect this observation to claims/figures/sections.
