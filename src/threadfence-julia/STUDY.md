# threadfence: verified four-model study observation

RQ IDs: RQ1–RQ5. Experiment IDs: E1–E5. Evidence ID: 20261016T234046748427Z_threadfence.
Configuration contract: `{"n": 4096, "internal_repeats": 10, "input": "allFloat32minusone", "outputs": "16blockpartial_sums(firstoverwrittentotal),UInt32completioncounter castFloat32", "timing": "mean complete reductionoperation withsync; excludesreset/setup/copy"}`.
Protocol: five_trials_last_four_v1, all five correctness checks retained, timing mean trials2–5.
Evidence: `/home/imo/HeCBench_julia/.porting-state/portable_julia/runs/20261016T234046748427Z_threadfence/result.json`, config.json, timings.csv, raw logs, binary artifacts.
Ontology: ontology/benchmarks/threadfence.yaml; SHA256 `2bfd10918040ff36bc775adfff5f8c8b5751de160b28e8fe1c47931715ceec62`. Original inferred ontology labels retained; no retroactive winner-based relabeling.

## Numerical confirmation
Each variant passes independent full-element CPU oracle checks and direct binary-array comparisons against native CUDA and CUDA.jl on every one of the five trials. Acceptance tolerances and all observed absolute/relative errors are preserved in result.json. NaN/nonfinite and shape/count checks are performed by the runner. Native device kernels are included unchanged; historical CUDA.jl original_main.jl and original_Makefile are retained. Parent compatibility entrypoints execute the historical original source. Study drivers change only explicitly recorded reference inputs/measurement handling, a separate phase from original translation.

Faithful native single-kernel last-block-completion reduction: perblockshared atomicpartial, volatileglobal partialstore, CUDA.threadfence, atomiccompletioncounter fetch-old, lastblock reduction andcounterreset. KA/JACC use frameworkworkgroup/barriers/Atomix plus explicit CUDA.threadfence and inlinevolatileglobalPTX loads/stores: hybrid_device_intrinsics backend-dependent, NOT pureportableimplementation. InstalledAtomixCUDA modify/replace ignoresordering argument, so acquire/release annotationalone wasnotacceptedasmemoryvisibilityproof. HistoricalCUDAjl studyreference preserves2-kerneltreealgorithm, explicitalgorithmicperformanceconfound. n4096 allminusone,r10, full16partialarray(firstoverwrittentotal) pluscompletioncounter0 verifiedexact everytrial, notjusttotalscalar. JACC shared8bytes receivesextra2globalzeroscopy vsnative/KAstaticshared, recordedmappingoverhead. Selectedfixture exactinteger-valuedFloat32, no randominput or widerdevice memorymodelclaim.

## Runtime: microseconds
Each trial records the synchronized launch-loop mean per internal repeat. This includes host dispatch and excludes JIT/compilation, setup, allocations, transfers, input reset, CPU checking and artifact writing. Each trial starts with the same inputs. A device warmup precedes trial1. All timing commands are serialized on GPU6 via the shared lock. Native CUDA, CUDA.jl, KA and JACC are measured under the same named contract. Julia environments pin CUDA runtime12.0; exact package/hardware evidence lives in setup and run bundles.

| Model | Trials1–5 (us) | Mean2–5 (us) |
|---|---|---|
| cuda | 20.6542, 20.7952, 20.7182, 20.6942, 20.6472 | 20.7137 |
| cuda_jl | 9.2561, 11.6851, 8.6641, 8.4570, 8.8021 | 9.4021 |
| ka | 35.7123, 26.8613, 22.6182, 22.4502, 22.2412 | 23.5427 |
| jacc | 30.1313, 29.2853, 25.6192, 25.3092, 25.2823 | 26.3740 |

## Attempts, tokens and feature support
Each new framework has one source candidate; zero compiler-feedback code repairs. Actual runner build/JIT invocations are recorded separately from timing trials; tool/reuse/setup failures are infrastructure events, not translation candidates. A five-trial timing series is not five translation attempts. Token accounting is partial source-text characters/4, with unavailable conversation/tool/reasoning/cached usage explicitly unknown; these estimates do not establish total billed cost or an abstraction cost ranking. Shared infrastructure/reference correction/paper work are separate phases. KA-first and JACC reuse make this incremental production effort, not independent translations. No unsupported framework feature was observed for this selected configuration.

## Ontology explanation and paper follow-up
Use the existing operation/access classifications as grouping evidence, with implementation choices above recorded separately. Any runtime gap is an observation. Potential launch/configuration overhead, memory access behavior, compiler output and changed specialization are competing explanations, not demonstrated causes. No measured memory traffic, occupancy or roofline position was collected in this pass. Profiling and size scaling remain pending; this small configuration cannot support broad abstraction-selection recommendations. RQ1 is answered for this contract; RQ2 cost completeness is partial; RQ3 has matched measurements; RQ4 mechanisms and RQ5 generalization remain unresolved. Coordinator must export a portable snapshot and connect this observation to claims/figures/sections.
