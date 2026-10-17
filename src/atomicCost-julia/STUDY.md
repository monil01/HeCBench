# atomicCost: verified four-model study observation

RQ IDs: RQ1–RQ5. Experiment IDs: E1–E5. Evidence ID: 20261016T230615796265Z_atomicCost.
Configuration contract: `{"precision": ["Float64", "Int32", "Float32"], "num_threads": 4096, "elements_per_thread": 4, "total_implicit_elements": 16384, "internal_repeats": 10, "fixture": "implicitvaluei%2; eachthreadownoutputinitial0; nointerthreadcontention", "cases": ["atomic", "non-atomic"], "outputs": "6complete4096elementarrays convertedexactlytoFloat64", "timing": "sum6separatelysynchronizedloopmeans"}`.
Protocol: five_trials_last_four_v1, all five correctness checks retained, timing mean trials2–5.
Evidence: `/home/imo/HeCBench_julia/.porting-state/portable_julia/runs/20261016T230615796265Z_atomicCost/result.json`, config.json, timings.csv, raw logs, binary artifacts.
Ontology: ontology/benchmarks/atomicCost.yaml; SHA256 `0cca532634802b6c004be8e33ab79f840450d91d2abfbfa8ca1b9f8763620212`. Original inferred ontology labels retained; no retroactive winner-based relabeling.

## Numerical confirmation
Each variant passes independent full-element CPU oracle checks and direct binary-array comparisons against native CUDA and CUDA.jl on every one of the five trials. Acceptance tolerances and all observed absolute/relative errors are preserved in result.json. NaN/nonfinite and shape/count checks are performed by the runner. Native device kernels are included unchanged; historical CUDA.jl original_main.jl and original_Makefile are retained. Parent compatibility entrypoints execute the historical original source. Study drivers record reference input alignment, any algorithm correction, and measurement handling as separate phases from the historical translation.

Allatomic/non-atomic cases acrossFloat64,Int32,Float32 arepresent,4096threadoutputs each,24576completevalues checkedexactlyall5bothrefs. Namedstudy implicitlength16384 (4096threads times4elements) replacesnativedefault922521600. Everythread operatesonitsownaddress: this measuresatomicinstructionoverhead withoutinterthreadaddresscontention. Perthreadsum ofimplicit i%2 values ischeckedanalytically, notjustatomic/non-atomic equality. HistoricalCUDA.jl non-atomicpath accumulatesinregister thenstores once; nativeperitem+= maycompilercoalesce, whileKA/JACC preserveperitemupdate. Treatthisasimplementationconfound. Runtime sums6percase means; rawsubkernel times retained.

## Runtime: microseconds
Five in-process trials are retained; the reported mean excludes trial1 and averages trials2–5. Each value uses the configuration's exact timed region and normalization. Notes above and `input_contract` distinguish per-repeat means from case or pipeline totals. Transfers, resets, allocation, temporary storage, graph control, host dispatch and synchronization are included where that named contract specifies. Compiler/JIT warmup, CPU correctness and artifact writing occur outside measurement. All GPU measurements use the shared GPU6 lock. Exact package/hardware evidence lives in setup and run bundles.

| Model | Trials1–5 (us) | Mean2–5 (us) |
|---|---|---|
| cuda | 20.3884, 18.9381, 19.0611, 19.2881, 19.0912 | 19.0946 |
| cuda_jl | 33.8284, 29.1390, 27.1574, 27.0504, 26.7374 | 27.5211 |
| ka | 56.2465, 58.0445, 41.6074, 40.3973, 38.8484 | 44.7244 |
| jacc | 92.8038, 64.3975, 55.9856, 53.2656, 53.4176 | 56.7666 |

## Attempts, tokens and feature support
Each new framework has one source candidate; zero compiler-feedback code repairs. Actual runner build/JIT invocations are recorded separately from timing trials; tool/reuse/setup failures are infrastructure events, not translation candidates. A five-trial timing series is not five translation attempts. Token accounting is partial source-text characters/4, with unavailable conversation/tool/reasoning/cached usage explicitly unknown; these estimates do not establish total billed cost or an abstraction cost ranking. Shared infrastructure/reference correction/paper work are separate phases. KA-first and JACC reuse make this incremental production effort, not independent translations. No unsupported framework feature was observed for this selected configuration.

## Ontology explanation and paper follow-up
Use the existing operation/access classifications as grouping evidence, with implementation choices above recorded separately. Any runtime gap is an observation. Potential launch/configuration overhead, memory access behavior, compiler output and changed specialization are competing explanations, not demonstrated causes. No measured memory traffic, occupancy or roofline position was collected in this pass. Profiling and size scaling remain pending; this small configuration cannot support broad abstraction-selection recommendations. RQ1 is answered for this contract; RQ2 cost completeness is partial; RQ3 has matched measurements; RQ4 mechanisms and RQ5 generalization remain unresolved. Coordinator must export a portable snapshot and connect this observation to claims/figures/sections.
