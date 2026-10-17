# atomicAggregate: verified four-model study observation

RQ IDs: RQ1–RQ5. Experiment IDs: E1–E5. Evidence ID: 20261016T225909452541Z_atomicAggregate.
Configuration contract: `{"precision": "Int32", "blocks": 64, "threads_per_block": 256, "internal_repeats": 10, "locations": [32, 16, 8, 4, 2, 1], "fixture": "same-key groups threadlocalindex%s; outputinitialzero", "outputs": "63resultcounts concatenated6locationcases", "timing": "sum6synchronizedloopmeans"}`.
Protocol: five_trials_last_four_v1, all five correctness checks retained, timing mean trials2–5.
Evidence: `/home/imo/HeCBench_julia/.porting-state/portable_julia/runs/20261016T225909452541Z_atomicAggregate/result.json`, config.json, timings.csv, raw logs, binary artifacts.
Ontology: ontology/benchmarks/atomicAggregate.yaml; SHA256 `a055c471c0bb26998b8e0592705c4748c351dc81294ebbdb0d3142f538561bbb`. Original inferred ontology labels retained; no retroactive winner-based relabeling.

## Numerical confirmation
Each variant passes independent full-element CPU oracle checks and direct binary-array comparisons against native CUDA and CUDA.jl on every one of the five trials. Acceptance tolerances and all observed absolute/relative errors are preserved in result.json. NaN/nonfinite and shape/count checks are performed by the runner. Native device kernels are included unchanged; historical CUDA.jl original_main.jl and original_Makefile are retained. Parent compatibility entrypoints execute the historical original source. Study drivers record reference input alignment, any algorithm correction, and measurement handling as separate phases from the historical translation.

Explicit PORTABLE ALGORITHM ADAPTATION: native __match_any_sync (or fallbackballot/shuffle) is replaced with sharedmemory32-peer keymatching andblockbarrier. Eachwarp stillselectsoneleader per same-addressgroup andperforms oneatomicAdd ofitspeercount, preservingaggregation ratherthanperthreadatomic fallback. Thehelperreturnbroadcast isomitted because the nativecallingkernel discardsit; this isobservablekernel equivalence, notgenerichelperreturn equivalence. KA/JACCload anextra256-element sharedkeyinput, nativecomputeskey onthefly. KA usesstaticlocalmemory; JACC.shared copies256keys into1024dynamicsharedbytes. All63outputcounts in6locationcases matchbothrefs exactlyall5. Thiscase measures thecost ofportablealgorithm adaptation, notequal-instruction abstractionoverhead; groupseparately fromdirectports. No sourceAPI search isreported asproof ofuniversal unsupportedfeatures.

## Runtime: microseconds
Five in-process trials are retained; the reported mean excludes trial1 and averages trials2–5. Each value uses the configuration's exact timed region and normalization. Notes above and `input_contract` distinguish per-repeat means from case or pipeline totals. Transfers, resets, allocation, temporary storage, graph control, host dispatch and synchronization are included where that named contract specifies. Compiler/JIT warmup, CPU correctness and artifact writing occur outside measurement. All GPU measurements use the shared GPU6 lock. Exact package/hardware evidence lives in setup and run bundles.

| Model | Trials1–5 (us) | Mean2–5 (us) |
|---|---|---|
| cuda | 22.5523, 23.6553, 22.6091, 22.5163, 22.5252 | 22.8265 |
| cuda_jl | 27.2502, 30.8453, 27.6503, 27.6633, 27.3723 | 28.3828 |
| ka | 62.3886, 67.5646, 63.0885, 62.5255, 62.4385 | 63.9043 |
| jacc | 47.0736, 56.9662, 47.2924, 47.1346, 48.0556 | 49.8622 |

## Attempts, tokens and feature support
Each new framework has one source candidate; zero compiler-feedback code repairs. Actual runner build/JIT invocations are recorded separately from timing trials; tool/reuse/setup failures are infrastructure events, not translation candidates. A five-trial timing series is not five translation attempts. Token accounting is partial source-text characters/4, with unavailable conversation/tool/reasoning/cached usage explicitly unknown; these estimates do not establish total billed cost or an abstraction cost ranking. Shared infrastructure/reference correction/paper work are separate phases. KA-first and JACC reuse make this incremental production effort, not independent translations. No unsupported framework feature was observed for this selected configuration.

## Ontology explanation and paper follow-up
Use the existing operation/access classifications as grouping evidence, with implementation choices above recorded separately. Any runtime gap is an observation. Potential launch/configuration overhead, memory access behavior, compiler output and changed specialization are competing explanations, not demonstrated causes. No measured memory traffic, occupancy or roofline position was collected in this pass. Profiling and size scaling remain pending; this small configuration cannot support broad abstraction-selection recommendations. RQ1 is answered for this contract; RQ2 cost completeness is partial; RQ3 has matched measurements; RQ4 mechanisms and RQ5 generalization remain unresolved. Coordinator must export a portable snapshot and connect this observation to claims/figures/sections.
