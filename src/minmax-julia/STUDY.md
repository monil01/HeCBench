# minmax: verified four-model study observation

RQ IDs: RQ1–RQ5. Experiment IDs: E1–E5. Evidence ID: 20261016T231917249887Z_minmax.
Configuration contract: `{"precision": "Float32", "size": 4096, "internal_repeats": 10, "fixture": "x=Float32(i*37%n), y=Float32(i*19%n)/Float32(n), i0based; uniqueextrema", "ordering": "squaredEuclideanmagnitude; noNaN; noextremalties", "cases": ["separatemin_and_max", "fusedminmax"], "outputs": "8Float32coordinates minx,miny,maxx,maxy percase", "timing": "sum2loopmeans INCLUDINGhostindex/result retrieval eachrepeat"}`.
Protocol: five_trials_last_four_v1, all five correctness checks retained, timing mean trials2–5.
Evidence: `/home/imo/HeCBench_julia/.porting-state/portable_julia/runs/20261016T231917249887Z_minmax/result.json`, config.json, timings.csv, raw logs, binary artifacts.
Ontology: ontology/benchmarks/minmax.yaml; SHA256 `77eaba52709aa50ea2a18dede0a8a9223380d7037dfbb13661a47c76c1cd9410`. Original inferred ontology labels retained; no retroactive winner-based relabeling.

## Numerical confirmation
Each variant passes independent full-element CPU oracle checks and direct binary-array comparisons against native CUDA and CUDA.jl on every one of the five trials. Acceptance tolerances and all observed absolute/relative errors are preserved in result.json. NaN/nonfinite and shape/count checks are performed by the runner. Native device kernels are included unchanged; historical CUDA.jl original_main.jl and original_Makefile are retained. Parent compatibility entrypoints execute the historical original source. Study drivers record reference input alignment, any algorithm correction, and measurement handling as separate phases from the historical translation.

All8pointcoordinates (minx,miny,maxx,maxy for2cases) checkedexactlyagainsthostextremascan andbothGPUreferences all5. Fixture4096permutedFloat32points hasuniqueextrema/noNaNs; tie andNaNbehavior untested. NativeusesThrust pointAoS comparator; CUDA.jl usesSoA temporarymagnitudes+argmin/argmax andduplicatesseparateoperation infusedslot ashistoricalimplementation. KAimplementsproper256threadsharedmemory2stagepointreduction withseparatemin/maxpasses versusfusedbothpass. JACCusespublicreducerAPI withcustomisbitsExtremum/ExtremaPair andpersistentworkspaces: separatechoose_min/choose_max reducers versusfusedmerge reducer. NoCUDAkernelfallback. Timedregionincludeshostretrieval ofindices/coordinates eachinternalrepeat matchingnativehigh-leveloperation, notkernel-onlythroughput. NativeThrustlibrarycache/strategy, AoSvsSoA, temporarybuffers andlaunchcount areimplementationexplanations toprofile separately. This demonstratescustom-stateJACCreduction support atpinnedversion; no unsupportedclaim.

## Runtime: microseconds
Five in-process trials are retained; the reported mean excludes trial1 and averages trials2–5. Each value uses the configuration's exact timed region and normalization. Notes above and `input_contract` distinguish per-repeat means from case or pipeline totals. Transfers, resets, allocation, temporary storage, graph control, host dispatch and synchronization are included where that named contract specifies. Compiler/JIT warmup, CPU correctness and artifact writing occur outside measurement. All GPU measurements use the shared GPU6 lock. Exact package/hardware evidence lives in setup and run bundles.

| Model | Trials1–5 (us) | Mean2–5 (us) |
|---|---|---|
| cuda | 155.9843, 154.5054, 153.6344, 154.9483, 154.2193 | 154.3269 |
| cuda_jl | 260.8962, 287.4805, 273.0994, 274.6094, 274.3094 | 277.3747 |
| ka | 143.1211, 155.5734, 144.5823, 139.5472, 139.3803 | 144.7708 |
| jacc | 223.4450, 193.7727, 181.4616, 175.8716, 174.8775 | 181.4959 |

## Attempts, tokens and feature support
Each new framework has one source candidate; zero compiler-feedback code repairs. Actual runner build/JIT invocations are recorded separately from timing trials; tool/reuse/setup failures are infrastructure events, not translation candidates. A five-trial timing series is not five translation attempts. Token accounting is partial source-text characters/4, with unavailable conversation/tool/reasoning/cached usage explicitly unknown; these estimates do not establish total billed cost or an abstraction cost ranking. Shared infrastructure/reference correction/paper work are separate phases. KA-first and JACC reuse make this incremental production effort, not independent translations. No unsupported framework feature was observed for this selected configuration.

## Ontology explanation and paper follow-up
Use the existing operation/access classifications as grouping evidence, with implementation choices above recorded separately. Any runtime gap is an observation. Potential launch/configuration overhead, memory access behavior, compiler output and changed specialization are competing explanations, not demonstrated causes. No measured memory traffic, occupancy or roofline position was collected in this pass. Profiling and size scaling remain pending; this small configuration cannot support broad abstraction-selection recommendations. RQ1 is answered for this contract; RQ2 cost completeness is partial; RQ3 has matched measurements; RQ4 mechanisms and RQ5 generalization remain unresolved. Coordinator must export a portable snapshot and connect this observation to claims/figures/sections.
