# lfib4: verified four-model study observation

RQ IDs: RQ1–RQ5. Experiment IDs: E1–E5. Evidence ID: 20261016T234202779904Z_lfib4.
Configuration contract: `{"n": 16384, "r": 16, "s": 1024, "seed": "256UInt32(i*1664525+1013904223) fori0..255", "outputs": "all16384UInt32modularrecurrencevalues", "timing": "fourstagepipeline +2streamjoins+finalsync; excludesallallocation/seedreset/transfers/streamcreation"}`.
Protocol: five_trials_last_four_v1, all five correctness checks retained, timing mean trials2–5.
Evidence: `/home/imo/HeCBench_julia/.porting-state/portable_julia/runs/20261016T234202779904Z_lfib4/result.json`, config.json, timings.csv, raw logs, binary artifacts.
Ontology: ontology/benchmarks/lfib4.yaml; SHA256 `0514fd2e41e2a35d730a5c9c562a6d66a96227c736b704a7ea4b99914d3f6f7d`. Original inferred ontology labels retained; no retroactive winner-based relabeling.

## Numerical confirmation
Each variant passes independent full-element CPU oracle checks and direct binary-array comparisons against native CUDA and CUDA.jl on every one of the five trials. Acceptance tolerances and all observed absolute/relative errors are preserved in result.json. NaN/nonfinite and shape/count checks are performed by the runner. Native device kernels are included unchanged; historical CUDA.jl original_main.jl and original_Makefile are retained. Parent compatibility entrypoints execute the historical original source. Study drivers change only explicitly recorded reference inputs/measurement handling, a separate phase from original translation.

Faithful native four-stage LFIB4parallelalgorithm implemented: firstcolumn andpolynomialcolumn runin2CUDAstreams, join, lastentriesconvolution, independentcolumnfill. UInt32modularP1=55 P2=119 P3=179 P4=256. HistoricalCUDAjl ONEthreadserialGPUreference isretainedoriginaland repairedinexplicitseparatereferencephase toactualparallel4stagekernel. All16384UInt32outputsexact independentserialCPUrecurrence andbothrefs everytrial. Namedr16s1024seedformula fixture only, notwholeoriginalr16..4096sweep. Timedregionpipeline +streamjoins/finalfence excludesoriginalgLFIB4 alloc/seedcopy/streamcreation/free uniformly4models. Frameworkkernels requireexplicitCUDAhoststreamcontrol => hybrid_host_control, portabledevicekernelsbutbackend-dependenthostorchestration. KAstaticshared versusJACC copies2048/3072/9216/16384bytes ofzerosfromglobalperstage. Initializeddeadnativeworkspace improvesdefinedexecutionwithoutchangingobservableoutput; overheadconfound. KA/JACCsourcecandidate1 succeed. RepairedCUDAjlreference firstcompilationB2 failedimplicitUInt32return fromcolumn_y!, explicitreturnnothing addedall4kernels (reference2candidates/2checks/1compilerfeedbackrepair).

## Runtime: microseconds
Each trial records the synchronized launch-loop mean per internal repeat. This includes host dispatch and excludes JIT/compilation, setup, allocations, transfers, input reset, CPU checking and artifact writing. Each trial starts with the same inputs. A device warmup precedes trial1. All timing commands are serialized on GPU6 via the shared lock. Native CUDA, CUDA.jl, KA and JACC are measured under the same named contract. Julia environments pin CUDA runtime12.0; exact package/hardware evidence lives in setup and run bundles.

| Model | Trials1–5 (us) | Mean2–5 (us) |
|---|---|---|
| cuda | 88.7500, 88.1010, 87.2710, 87.6810, 87.3610 | 87.6035 |
| cuda_jl | 225.5720, 225.8120, 205.8620, 201.0120, 201.2520 | 208.4845 |
| ka | 322.1030, 317.4430, 294.1230, 289.3720, 285.5030 | 296.6103 |
| jacc | 378.3430, 326.9430, 282.1120, 271.5230, 265.6620 | 286.5600 |

## Attempts, tokens and feature support
Each new framework has one source candidate; zero compiler-feedback code repairs. Actual runner build/JIT invocations are recorded separately from timing trials; tool/reuse/setup failures are infrastructure events, not translation candidates. A five-trial timing series is not five translation attempts. Token accounting is partial source-text characters/4, with unavailable conversation/tool/reasoning/cached usage explicitly unknown; these estimates do not establish total billed cost or an abstraction cost ranking. Shared infrastructure/reference correction/paper work are separate phases. KA-first and JACC reuse make this incremental production effort, not independent translations. No unsupported framework feature was observed for this selected configuration.

## Ontology explanation and paper follow-up
Use the existing operation/access classifications as grouping evidence, with implementation choices above recorded separately. Any runtime gap is an observation. Potential launch/configuration overhead, memory access behavior, compiler output and changed specialization are competing explanations, not demonstrated causes. No measured memory traffic, occupancy or roofline position was collected in this pass. Profiling and size scaling remain pending; this small configuration cannot support broad abstraction-selection recommendations. RQ1 is answered for this contract; RQ2 cost completeness is partial; RQ3 has matched measurements; RQ4 mechanisms and RQ5 generalization remain unresolved. Coordinator must export a portable snapshot and connect this observation to claims/figures/sections.
