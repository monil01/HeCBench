# relu: verified four-model study observation

RQ IDs: RQ1–RQ5. Experiment IDs: E1–E5. Evidence ID: 20261016T224701187590Z_relu.
Configuration contract: `{"size": 4096, "precision": "Float16 gradient and signedInt8 packedUInt32; outputs serialized exactly inFloat64", "fixture": "feature[i]=(zero_based_i%17-8)/8; gradient1; packed=UInt32(i*2654435761 modulo2^32)", "internal_repeats": 100, "outputs": "4subkernels concatenated in order ReluGrad_impl1,ReluGrad_impl2,Relu_impl1,Relu_impl2; nvalues each", "timing": "sum4 separately synchronized loop means perrepeat; rawsubkernel times retained"}`.
Protocol: five_trials_last_four_v1, all five correctness checks retained, timing mean trials2–5.
Evidence: `/home/imo/HeCBench_julia/.porting-state/portable_julia/runs/20261016T224701187590Z_relu/result.json`, config.json, timings.csv, raw logs, binary artifacts.
Ontology: ontology/benchmarks/relu.yaml; SHA256 `ae65f5b75c9aaa5392b103508cfb735e97522fb8534ab7a3956b9566c8f7a4c5`. Original inferred ontology labels retained; no retroactive winner-based relabeling.

## Numerical confirmation
Each variant passes independent full-element CPU oracle checks and direct binary-array comparisons against native CUDA and CUDA.jl on every one of the five trials. Acceptance tolerances and all observed absolute/relative errors are preserved in result.json. NaN/nonfinite and shape/count checks are performed by the runner. Native device kernels are included unchanged; historical CUDA.jl original_main.jl and original_Makefile are retained. Parent compatibility entrypoints execute the historical original source. Study drivers record reference input alignment, any algorithm correction, and measurement handling as separate phases from the historical translation.

Allfour native benchmark subkernels are included: half2 andhalf8 ReluGrad, packed-four-signed-byte scalar and__vmaxs4 Relu. Each trial retains16384values (4096percase), serialized exactly as Float64 without changing Float16 or UInt32 mathematical values. Native vectorization/intrinsic paths differ from historicalCUDA.jl and newKA/JACC scalar equivalent implementations. Runtime is SUM of the four separately synchronized perrepeat means; subkernel_times_seconds preserves eachcomponent. This is not one-kernel latency and does not establish a compiler vectorization mechanism; inspect percomponent evidence. No half/packed output value is skipped.

## Runtime: microseconds
Five in-process trials are retained; the reported mean excludes trial1 and averages trials2–5. Each value uses the configuration's exact timed region and normalization. Notes above and `input_contract` distinguish per-repeat means from case or pipeline totals. Transfers, resets, allocation, temporary storage, graph control, host dispatch and synchronization are included where that named contract specifies. Compiler/JIT warmup, CPU correctness and artifact writing occur outside measurement. All GPU measurements use the shared GPU6 lock. Exact package/hardware evidence lives in setup and run bundles.

| Model | Trials1–5 (us) | Mean2–5 (us) |
|---|---|---|
| cuda | 10.0928, 9.9881, 10.2403, 10.0009, 9.9615 | 10.0477 |
| cuda_jl | 10.3829, 10.2796, 10.2289, 10.1320, 10.1528 | 10.1983 |
| ka | 17.6084, 16.8465, 16.7181, 16.9988, 16.5970 | 16.7901 |
| jacc | 25.9256, 29.9978, 28.0447, 28.0344, 28.1749 | 28.5630 |

## Attempts, tokens and feature support
Each new framework has one source candidate; zero compiler-feedback code repairs. Actual runner build/JIT invocations are recorded separately from timing trials; tool/reuse/setup failures are infrastructure events, not translation candidates. A five-trial timing series is not five translation attempts. Token accounting is partial source-text characters/4, with unavailable conversation/tool/reasoning/cached usage explicitly unknown; these estimates do not establish total billed cost or an abstraction cost ranking. Shared infrastructure/reference correction/paper work are separate phases. KA-first and JACC reuse make this incremental production effort, not independent translations. No unsupported framework feature was observed for this selected configuration.

## Ontology explanation and paper follow-up
Use the existing operation/access classifications as grouping evidence, with implementation choices above recorded separately. Any runtime gap is an observation. Potential launch/configuration overhead, memory access behavior, compiler output and changed specialization are competing explanations, not demonstrated causes. No measured memory traffic, occupancy or roofline position was collected in this pass. Profiling and size scaling remain pending; this small configuration cannot support broad abstraction-selection recommendations. RQ1 is answered for this contract; RQ2 cost completeness is partial; RQ3 has matched measurements; RQ4 mechanisms and RQ5 generalization remain unresolved. Coordinator must export a portable snapshot and connect this observation to claims/figures/sections.
