# complex: verified four-model study observation

RQ IDs: RQ1–RQ5. Experiment IDs: E1–E5. Evidence ID: 20261016T230849866338Z_complex.
Configuration contract: `{"precision": ["Float32", "Float64"], "size": 4096, "internal_repeats": 10, "fixture": "NativeLCG seed1 fastforwardzeroindex; 4sequentialuniformvalues Float64 thenprecisioncast", "cases": "nativecustomFloatComplex/refcuComplex float,double; JuliausesComplexF32/F64 forbothslots", "outputs": "4*n integerpropertyscores(0..5); NOTrawcomplexvaluearrays", "acceptance": "5identities peritem eachresidual<1e-3; outputscoreall5", "timing": "sum4synchronizedloopmeans"}`.
Protocol: five_trials_last_four_v1, all five correctness checks retained, timing mean trials2–5.
Evidence: `/home/imo/HeCBench_julia/.porting-state/portable_julia/runs/20261016T230849866338Z_complex/result.json`, config.json, timings.csv, raw logs, binary artifacts.
Ontology: ontology/benchmarks/complex.yaml; SHA256 `876b1f5512a37b5977bf8ad149873e30dca888245ee1bb57747cbb49252d5fff`. Original inferred ontology labels retained; no retroactive winner-based relabeling.

## Numerical confirmation
Each variant passes independent full-element CPU oracle checks and direct binary-array comparisons against native CUDA and CUDA.jl on every one of the five trials. Acceptance tolerances and all observed absolute/relative errors are preserved in result.json. NaN/nonfinite and shape/count checks are performed by the runner. Native device kernels are included unchanged; historical CUDA.jl original_main.jl and original_Makefile are retained. Parent compatibility entrypoints execute the historical original source. Study drivers record reference input alignment, any algorithm correction, and measurement handling as separate phases from the historical translation.

EVIDENCESTRENGTH: completePROPERTY-SCORE arrays, NOTrawcomplexvaluearrays. Everythread emits score0..5 forfivecomplexidentities underresidualtolerance1e-3; all4*n=16384scores equal5 andmatchbothrefsall5. NativeCUDA customcomplex andcuComplexreferencefloat/double are4distinctpaths; JuliausesstandardComplexF32/F64 andduplicates2timing slots. HistoricalJulia sin/cos-generatedinputs differedfromnativeLCG; newstudydrivers alignexactnativeUInt64fastforwardLCG seed1, zero-basedthreadindex,mod2^63 and4randomvalues (Float64thencast). Inputgenerationrepair separatefromtranslation. Allscorespassingisnotsufficienttoshowrawfloating-pointoperation equivalence orprecisionerrorbounds. Treatasnativepropertycheck-strength despitefullartifactscorecomparison. Runtimes sum4percase means,sourceLCGworkincluded.

## Runtime: microseconds
Five in-process trials are retained; the reported mean excludes trial1 and averages trials2–5. Each value uses the configuration's exact timed region and normalization. Notes above and `input_contract` distinguish per-repeat means from case or pipeline totals. Transfers, resets, allocation, temporary storage, graph control, host dispatch and synchronization are included where that named contract specifies. Compiler/JIT warmup, CPU correctness and artifact writing occur outside measurement. All GPU measurements use the shared GPU6 lock. Exact package/hardware evidence lives in setup and run bundles.

| Model | Trials1–5 (us) | Mean2–5 (us) |
|---|---|---|
| cuda | 22.4260, 22.1712, 22.0592, 22.7832, 21.8701 | 22.2209 |
| cuda_jl | 26.8760, 31.6902, 27.3433, 26.6393, 26.7394 | 28.1031 |
| ka | 32.4174, 47.8084, 32.2314, 31.2211, 33.9154 | 36.2941 |
| jacc | 68.4406, 96.4497, 65.9166, 66.4746, 65.4766 | 73.5794 |

## Attempts, tokens and feature support
Each new framework has one source candidate; zero compiler-feedback code repairs. Actual runner build/JIT invocations are recorded separately from timing trials; tool/reuse/setup failures are infrastructure events, not translation candidates. A five-trial timing series is not five translation attempts. Token accounting is partial source-text characters/4, with unavailable conversation/tool/reasoning/cached usage explicitly unknown; these estimates do not establish total billed cost or an abstraction cost ranking. Shared infrastructure/reference correction/paper work are separate phases. KA-first and JACC reuse make this incremental production effort, not independent translations. No unsupported framework feature was observed for this selected configuration.

## Ontology explanation and paper follow-up
Use the existing operation/access classifications as grouping evidence, with implementation choices above recorded separately. Any runtime gap is an observation. Potential launch/configuration overhead, memory access behavior, compiler output and changed specialization are competing explanations, not demonstrated causes. No measured memory traffic, occupancy or roofline position was collected in this pass. Profiling and size scaling remain pending; this small configuration cannot support broad abstraction-selection recommendations. RQ1 is answered for this contract; RQ2 cost completeness is partial; RQ3 has matched measurements; RQ4 mechanisms and RQ5 generalization remain unresolved. Coordinator must export a portable snapshot and connect this observation to claims/figures/sections.
