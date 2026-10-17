# flip: verified four-model study observation

RQ IDs: RQ1–RQ5. Experiment IDs: E1–E5. Evidence ID: 20261016T224429604371Z_flip.
Configuration contract: `{"precision": "float32", "size": 4096, "internal_repeats": 100, "fixture": "3D16x16x16 rowmajor contiguous; input0:4095, allthreeaxes reversed"}`.
Protocol: five_trials_last_four_v1, all five correctness checks retained, timing mean trials2–5.
Evidence: `/home/imo/HeCBench_julia/.porting-state/portable_julia/runs/20261016T224429604371Z_flip/result.json`, config.json, timings.csv, raw logs, binary artifacts.
Ontology: ontology/benchmarks/flip.yaml; SHA256 `a9ed4639846ea00602e117b2a82e30e41f66eb6a280b95142b1c895b39c96851`. Original inferred ontology labels retained; no retroactive winner-based relabeling.

## Numerical confirmation
Each variant passes independent full-element CPU oracle checks and direct binary-array comparisons against native CUDA and CUDA.jl on every one of the five trials. Acceptance tolerances and all observed absolute/relative errors are preserved in result.json. NaN/nonfinite and shape/count checks are performed by the runner. Native device kernels are included unchanged; historical CUDA.jl original_main.jl and original_Makefile are retained. Parent compatibility entrypoints execute the historical original source. Study drivers record reference input alignment, any algorithm correction, and measurement handling as separate phases from the historical translation.

This named study selects FP32 contiguous3D16cubed input and reverses allthreeaxes. The original benchmark also exercisesFP64; noFP64 timing/correctness claim is made here. Native kernel preserves generic index/stride/shape loops withGPU metadata arrays; historical CUDA.jl and newports specialize cubic3D mapping using scalar dimension arguments. That specialization is an implementation confound. Deterministic input0:4095 is checked exactly for everyelement everytrial.

## Runtime: microseconds
Five in-process trials are retained; the reported mean excludes trial1 and averages trials2–5. Each value uses the configuration's exact timed region and normalization. Notes above and `input_contract` distinguish per-repeat means from case or pipeline totals. Transfers, resets, allocation, temporary storage, graph control, host dispatch and synchronization are included where that named contract specifies. Compiler/JIT warmup, CPU correctness and artifact writing occur outside measurement. All GPU measurements use the shared GPU6 lock. Exact package/hardware evidence lives in setup and run bundles.

| Model | Trials1–5 (us) | Mean2–5 (us) |
|---|---|---|
| cuda | 4.2292, 4.2393, 4.2446, 4.2385, 4.2248 | 4.2368 |
| cuda_jl | 3.0705, 3.0850, 2.7742, 2.7520, 2.7523 | 2.8409 |
| ka | 5.4819, 5.0656, 4.8699, 4.6834, 4.5275 | 4.7866 |
| jacc | 7.7565, 7.5505, 7.2260, 7.0935, 6.9462 | 7.2040 |

## Attempts, tokens and feature support
Each new framework has one source candidate; zero compiler-feedback code repairs. Actual runner build/JIT invocations are recorded separately from timing trials; tool/reuse/setup failures are infrastructure events, not translation candidates. A five-trial timing series is not five translation attempts. Token accounting is partial source-text characters/4, with unavailable conversation/tool/reasoning/cached usage explicitly unknown; these estimates do not establish total billed cost or an abstraction cost ranking. Shared infrastructure/reference correction/paper work are separate phases. KA-first and JACC reuse make this incremental production effort, not independent translations. No unsupported framework feature was observed for this selected configuration.

## Ontology explanation and paper follow-up
Use the existing operation/access classifications as grouping evidence, with implementation choices above recorded separately. Any runtime gap is an observation. Potential launch/configuration overhead, memory access behavior, compiler output and changed specialization are competing explanations, not demonstrated causes. No measured memory traffic, occupancy or roofline position was collected in this pass. Profiling and size scaling remain pending; this small configuration cannot support broad abstraction-selection recommendations. RQ1 is answered for this contract; RQ2 cost completeness is partial; RQ3 has matched measurements; RQ4 mechanisms and RQ5 generalization remain unresolved. Coordinator must export a portable snapshot and connect this observation to claims/figures/sections.
