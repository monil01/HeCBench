# atomicPerf: verified four-model study observation

RQ IDs: RQ1–RQ5. Experiment IDs: E1–E5. Evidence ID: 20261016T225930840699Z_atomicPerf.
Configuration contract: `{"precision": ["Float64", "Int32", "Float32"], "size": 4096, "output_buffer_size": 1024, "initial": "1:1024", "cases": ["BlockRangeGlobal", "WarpRangeGlobal", "SingleRangeGlobal"], "internal_repeats": 10, "singleoffset": "repeatindex0based%256", "outputs": "9216 values all9 completebuffers inFloat64 exactconversion", "timing": "sum9synchronizedkernel-loopmeans", "excluded_native_cases": {"shared_block_warp_single": "uninitialized shared storage; impossible blockIdx.x==gridDim.x outputguard, maycompileaway; invalidperformance reference"}}`.
Protocol: five_trials_last_four_v1, all five correctness checks retained, timing mean trials2–5.
Evidence: `/home/imo/HeCBench_julia/.porting-state/portable_julia/runs/20261016T225930840699Z_atomicPerf/result.json`, config.json, timings.csv, raw logs, binary artifacts.
Ontology: ontology/benchmarks/atomicPerf.yaml; SHA256 `5d6ba531a5427bc69b6f01914c46ce1e90a96761f6eded4017f6c4f6a2f88b0e`. Original inferred ontology labels retained; no retroactive winner-based relabeling.

## Numerical confirmation
Each variant passes independent full-element CPU oracle checks and direct binary-array comparisons against native CUDA and CUDA.jl on every one of the five trials. Acceptance tolerances and all observed absolute/relative errors are preserved in result.json. NaN/nonfinite and shape/count checks are performed by the runner. Native device kernels are included unchanged; historical CUDA.jl original_main.jl and original_Makefile are retained. Parent compatibility entrypoints execute the historical original source. Study drivers record reference input alignment, any algorithm correction, and measurement handling as separate phases from the historical translation.

NAMED GLOBAL-ONLY SUBSET: all3validglobaladdresscontention cases (blockrange,warprange,singleoffset) acrossFloat64,Int32,Float32 areimplemented,1024outputs percase,9216values totalall5. The3native sharedmemorycases areEXCLUDED fromstudyperformance: uninitializedsharedarrays plusimpossible outputguard blockIdx.x==gridDim.x mean noobservableoutput andpossiblecompilerelimination. HistoricalJulia usedzero-work sharedcases; preservedbyteforbyte. Thisis invalidnative-reference behavior, notframeworkunsupported. Seeexcluded_reference_cases.json. Report subsetcoverage ratherthanfulloriginal18casebenchmark completion. JACCconstructsLaunchSpec perlaunch, includedinhost-loopcost andnotoptimizedaway inthisfirsttranslation. No claimingmeasuredsharedatomicperformance.

## Runtime: microseconds
Five in-process trials are retained; the reported mean excludes trial1 and averages trials2–5. Each value uses the configuration's exact timed region and normalization. Notes above and `input_contract` distinguish per-repeat means from case or pipeline totals. Transfers, resets, allocation, temporary storage, graph control, host dispatch and synchronization are included where that named contract specifies. Compiler/JIT warmup, CPU correctness and artifact writing occur outside measurement. All GPU measurements use the shared GPU6 lock. Exact package/hardware evidence lives in setup and run bundles.

| Model | Trials1–5 (us) | Mean2–5 (us) |
|---|---|---|
| cuda | 45.5774, 45.5864, 45.3835, 45.3163, 45.5314 | 45.4544 |
| cuda_jl | 65.1065, 69.2025, 58.2875, 55.4305, 55.9625 | 59.7208 |
| ka | 175.6256, 152.8876, 148.3902, 176.2455, 170.1776 | 161.9252 |
| jacc | 167.2596, 132.9861, 118.6041, 98.6100, 96.8779 | 111.7695 |

## Attempts, tokens and feature support
Each new framework has one source candidate; zero compiler-feedback code repairs. Actual runner build/JIT invocations are recorded separately from timing trials; tool/reuse/setup failures are infrastructure events, not translation candidates. A five-trial timing series is not five translation attempts. Token accounting is partial source-text characters/4, with unavailable conversation/tool/reasoning/cached usage explicitly unknown; these estimates do not establish total billed cost or an abstraction cost ranking. Shared infrastructure/reference correction/paper work are separate phases. KA-first and JACC reuse make this incremental production effort, not independent translations. No unsupported framework feature was observed for this selected configuration.

## Ontology explanation and paper follow-up
Use the existing operation/access classifications as grouping evidence, with implementation choices above recorded separately. Any runtime gap is an observation. Potential launch/configuration overhead, memory access behavior, compiler output and changed specialization are competing explanations, not demonstrated causes. No measured memory traffic, occupancy or roofline position was collected in this pass. Profiling and size scaling remain pending; this small configuration cannot support broad abstraction-selection recommendations. RQ1 is answered for this contract; RQ2 cost completeness is partial; RQ3 has matched measurements; RQ4 mechanisms and RQ5 generalization remain unresolved. Coordinator must export a portable snapshot and connect this observation to claims/figures/sections.
