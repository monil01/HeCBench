# atomicReduction: verified four-model study observation

RQ IDs: RQ1–RQ5. Experiment IDs: E1–E5. Evidence ID: 20261016T225345083749Z_atomicReduction.
Configuration contract: `{"precision": "Int32", "size": 65536, "fixture": "inputzeroindex%2", "block_sizes": [128, 256, 512, 1024], "widths": [1, 2, 4, 8, 16], "grid": "max(min(ceil(n/blocksize),2048)/width,1)", "internal_repeats": 10, "outputs": "20exact reductionscalars,expected32768each", "timing": "sum20 perrepeat means INCLUDING each outputreset, matchingnative loop"}`.
Protocol: five_trials_last_four_v1, all five correctness checks retained, timing mean trials2–5.
Evidence: `/home/imo/HeCBench_julia/.porting-state/portable_julia/runs/20261016T225345083749Z_atomicReduction/result.json`, config.json, timings.csv, raw logs, binary artifacts.
Ontology: ontology/benchmarks/atomicReduction.yaml; SHA256 `1cb1aac59305537e1d54be38f247abb3a64aa7793e015ffb99f4263fd9a8e566`. Original inferred ontology labels retained; no retroactive winner-based relabeling.

## Numerical confirmation
Each variant passes independent full-element CPU oracle checks and direct binary-array comparisons against native CUDA and CUDA.jl on every one of the five trials. Acceptance tolerances and all observed absolute/relative errors are preserved in result.json. NaN/nonfinite and shape/count checks are performed by the runner. Native device kernels are included unchanged; historical CUDA.jl original_main.jl and original_Makefile are retained. Parent compatibility entrypoints execute the historical original source. Study drivers record reference input alignment, any algorithm correction, and measurement handling as separate phases from the historical translation.

All20 native configurations are retained: block sizes128/256/512/1024 and element widths1/2/4/8/16. Input65536Int32 alternating0/1 gives exactchecksum32768; eachcompleteoutput reduction scalar is verified for everycase everytrial, notjusta printedPASS. The20scalar artifact is thecomplete mathematicaloutput, notexhaustive inspection ofall threadintermediates. Native5specialized kernels are unchanged. ExistingJulia andKA/JACC generic runtimewidth loop differs fromnativecompiletimeunrolling. The timedregion INCLUDES outputzeroing eachinternalrepeat, asnative; CUDAcudaMemset andJuliafill! maygenerate differentresetoperations, a confound recordedexplicitly. Runtime sums20percase means; raw20componenttimes retained.

## Runtime: microseconds
Five in-process trials are retained; the reported mean excludes trial1 and averages trials2–5. Each value uses the configuration's exact timed region and normalization. Notes above and `input_contract` distinguish per-repeat means from case or pipeline totals. Transfers, resets, allocation, temporary storage, graph control, host dispatch and synchronization are included where that named contract specifies. Compiler/JIT warmup, CPU correctness and artifact writing occur outside measurement. All GPU measurements use the shared GPU6 lock. Exact package/hardware evidence lives in setup and run bundles.

| Model | Trials1–5 (us) | Mean2–5 (us) |
|---|---|---|
| cuda | 125.9190, 123.8821, 124.1161, 122.2500, 125.0393 | 123.8219 |
| cuda_jl | 137.1972, 144.6724, 133.4074, 132.9421, 133.4831 | 136.1262 |
| ka | 234.2444, 232.4486, 221.4023, 219.5673, 219.1352 | 223.1384 |
| jacc | 210.7400, 224.7823, 194.8708, 194.5458, 192.8570 | 201.7640 |

## Attempts, tokens and feature support
Each new framework has one source candidate; zero compiler-feedback code repairs. Actual runner build/JIT invocations are recorded separately from timing trials; tool/reuse/setup failures are infrastructure events, not translation candidates. A five-trial timing series is not five translation attempts. Token accounting is partial source-text characters/4, with unavailable conversation/tool/reasoning/cached usage explicitly unknown; these estimates do not establish total billed cost or an abstraction cost ranking. Shared infrastructure/reference correction/paper work are separate phases. KA-first and JACC reuse make this incremental production effort, not independent translations. No unsupported framework feature was observed for this selected configuration.

## Ontology explanation and paper follow-up
Use the existing operation/access classifications as grouping evidence, with implementation choices above recorded separately. Any runtime gap is an observation. Potential launch/configuration overhead, memory access behavior, compiler output and changed specialization are competing explanations, not demonstrated causes. No measured memory traffic, occupancy or roofline position was collected in this pass. Profiling and size scaling remain pending; this small configuration cannot support broad abstraction-selection recommendations. RQ1 is answered for this contract; RQ2 cost completeness is partial; RQ3 has matched measurements; RQ4 mechanisms and RQ5 generalization remain unresolved. Coordinator must export a portable snapshot and connect this observation to claims/figures/sections.
