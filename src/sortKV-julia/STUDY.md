# sortKV: verified four-model study observation

RQ IDs: RQ1–RQ5. Experiment IDs: E1–E5. Evidence ID: 20261017T002307864697Z_sortKV.
Configuration contract: `{"n": 1024, "repeat": 3, "value_types": ["UInt8", "Int16", "Int32", "Int64"], "keys": "Int32((i2053)%n), uniquepermutation forpoweroftwon", "values": "T(key%256)", "outputs": "Allsortedkeysandvalues foreach4types exactFloat64encoding", "timing": "sum4endtoendH2D+sort+D2Hmeans, mainbufferspreallocated, algorithmscratchincluded"}`.
Protocol: five_trials_last_four_v1, all five correctness checks retained, timing mean trials2–5.
Evidence: `/home/imo/HeCBench_julia/.porting-state/portable_julia/runs/20261017T002307864697Z_sortKV/result.json`, config.json, timings.csv, raw logs, binary artifacts.
Ontology: ontology/benchmarks/sortKV.yaml; SHA256 `90460b156003bc87225b436bd3b6fe4aef7352f8dd2fc14e8ce981d953bcb226`. Original inferred ontology labels retained; no retroactive winner-based relabeling.

## Numerical confirmation
Each variant passes independent full-element CPU oracle checks and direct binary-array comparisons against native CUDA and CUDA.jl on every one of the five trials. Acceptance tolerances and all observed absolute/relative errors are preserved in result.json. NaN/nonfinite and shape/count checks are performed by the runner. Native device kernels are included unchanged; historical CUDA.jl original_main.jl and original_Makefile are retained. Parent compatibility entrypoints execute the historical original source. Study drivers record reference input alignment, any algorithm correction, and measurement handling as separate phases from the historical translation.

The study retains all four value types UInt8/Int16/Int32/Int64, Int32 keys and their key%256 associations. A shared affine unique-key permutation replaces incompatible RNG shuffles. All8192 sorted keys/values match CPU, native Thrust and CUDA.jl exactly in every trial atn1024,r3. Native sort_by_key, CUDA.jl sortperm+gather and authored KA/JACC55-stage bitonic networks use different algorithms; this is an explicit algorithm adaptation and cannot isolate abstraction overhead. Runtime includes H2D input copies, sorting (including algorithm-owned scratch/permutation allocations), D2H output copies and completion. Main buffers are preallocated uniformly, unlike the original native first-iteration allocation followed by capacity reuse. Only power-of-two unique keys are covered; duplicates, arbitrary lengths and NaNs are untested. Both frameworks succeed on candidate one with four type specializations. JACC reserves zero shared bytes.

## Runtime: microseconds
Five in-process trials are retained; the reported mean excludes trial1 and averages trials2–5. Each value uses the configuration's exact timed region and normalization. Notes above and `input_contract` distinguish per-repeat means from case or pipeline totals. Transfers, resets, allocation, temporary storage, graph control, host dispatch and synchronization are included where that named contract specifies. Compiler/JIT warmup, CPU correctness and artifact writing occur outside measurement. All GPU measurements use the shared GPU6 lock. Exact package/hardware evidence lives in setup and run bundles.

| Model | Trials1–5 (us) | Mean2–5 (us) |
|---|---|---|
| cuda | 290.7027, 283.1090, 283.5387, 289.6627, 282.7960 | 284.7766 |
| cuda_jl | 1424.8387, 1070.7723, 931.3643, 920.5780, 933.1847 | 963.9748 |
| ka | 1208.3373, 1168.7067, 1137.7497, 1152.8600, 1148.2730 | 1151.8973 |
| jacc | 2162.1257, 1807.5193, 1749.0857, 1726.2213, 1712.1053 | 1748.7329 |

## Attempts, tokens and feature support
Each new framework has one source candidate; zero compiler-feedback code repairs. Actual runner build/JIT invocations are recorded separately from timing trials; tool/reuse/setup failures are infrastructure events, not translation candidates. A five-trial timing series is not five translation attempts. Token accounting is partial source-text characters/4, with unavailable conversation/tool/reasoning/cached usage explicitly unknown; these estimates do not establish total billed cost or an abstraction cost ranking. Shared infrastructure/reference correction/paper work are separate phases. KA-first and JACC reuse make this incremental production effort, not independent translations. No unsupported framework feature was observed for this selected configuration.

## Ontology explanation and paper follow-up
Use the existing operation/access classifications as grouping evidence, with implementation choices above recorded separately. Any runtime gap is an observation. Potential launch/configuration overhead, memory access behavior, compiler output and changed specialization are competing explanations, not demonstrated causes. No measured memory traffic, occupancy or roofline position was collected in this pass. Profiling and size scaling remain pending; this small configuration cannot support broad abstraction-selection recommendations. RQ1 is answered for this contract; RQ2 cost completeness is partial; RQ3 has matched measurements; RQ4 mechanisms and RQ5 generalization remain unresolved. Coordinator must export a portable snapshot and connect this observation to claims/figures/sections.
