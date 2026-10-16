# atan2: verified four-model study observation

RQ IDs: RQ1–RQ5. Experiment IDs: E1–E5. Evidence ID: 20261016T235332514769Z_atan2.
Configuration contract: `{"n": 4096, "repeat": 10, "fixture": "x4+(i%1024)/256 y.25+((i37)%1024)/1024; positive smallangleavoidsinteger-sumoverflow", "outputs": "AllFloat32/Int32/Int16 outputs concatenatedFloat64 exactencoding", "floating_point": "Native --fmad=false, degree15mixeddoubleliteral groupingpreserved", "timing": "sum3case synchronizedloopmean"}`.
Protocol: five_trials_last_four_v1, all five correctness checks retained, timing mean trials2–5.
Evidence: `/home/imo/HeCBench_julia/.porting-state/portable_julia/runs/20261016T235332514769Z_atan2/result.json`, config.json, timings.csv, raw logs, binary artifacts.
Ontology: ontology/benchmarks/atan2.yaml; SHA256 `5e05c4a9836400ab5de3404ccc12904de332c88cb0ef8bfa23b94a8ed1ff54df`. Original inferred ontology labels retained; no retroactive winner-based relabeling.

## Numerical confirmation
Each variant passes independent full-element CPU oracle checks and direct binary-array comparisons against native CUDA and CUDA.jl on every one of the five trials. Acceptance tolerances and all observed absolute/relative errors are preserved in result.json. NaN/nonfinite and shape/count checks are performed by the runner. Native device kernels are included unchanged; historical CUDA.jl original_main.jl and original_Makefile are retained. Parent compatibility entrypoints execute the historical original source. Study drivers change only explicitly recorded reference inputs/measurement handling, a separate phase from original translation.

All3nativekernels Float32sumof7polynomials, Int32sumof7, Int16sumof4 retained, all12288values EXACTCPU+dualrefs everytrial. Namedpositive-smallangle x4..8,y.25..1.25 fixture avoids nativeInt32signedsumoverflow presentinoriginalpositive1.57..2.57range; notclaimedoriginaldefaultrandominput. Nozeros/negativequadrants/extremes tested. Native--fmad=false explicitnoncontractedconfiguration. Degree15nativefloat(hexdoubledliteral+Float32tail) mixedprecisiongrouping repairedseparatelyinreference; historicalJuliaallFloat32Horner differed. KAinitialcandidateexcessparenthesis failedparserB2 bundle234801031878Z, candidate2 splitsintail/inner statements andpasses234939282522Z. Repairedreferencehasthesamesyntaxrepair initsseparatephase. JACCinheritscorrectedhelperscandidate1, nofeedbackrepair. Time sum3kernelcase means andseparatecasetimes retained.

## Runtime: microseconds
Each trial records the synchronized launch-loop mean per internal repeat. This includes host dispatch and excludes JIT/compilation, setup, allocations, transfers, input reset, CPU checking and artifact writing. Each trial starts with the same inputs. A device warmup precedes trial1. All timing commands are serialized on GPU6 via the shared lock. Native CUDA, CUDA.jl, KA and JACC are measured under the same named contract. Julia environments pin CUDA runtime12.0; exact package/hardware evidence lives in setup and run bundles.

| Model | Trials1–5 (us) | Mean2–5 (us) |
|---|---|---|
| cuda | 9.6201, 9.6150, 9.4690, 9.3691, 9.3582 | 9.4528 |
| cuda_jl | 15.0112, 21.8312, 12.6782, 13.2220, 12.6241 | 15.0889 |
| ka | 26.4873, 28.3974, 20.8062, 19.7300, 19.8282 | 22.1905 |
| jacc | 39.3054, 39.7703, 29.6833, 27.7132, 26.6631 | 30.9575 |

## Attempts, tokens and feature support
KA has two source candidates/two compiler checks/one B2 parser repair; JACC has one candidate/one check/zero feedback repairs. Actual runner build/JIT invocations are recorded separately from timing trials; tool/reuse/setup failures are infrastructure events, not translation candidates. A five-trial timing series is not five translation attempts. Token accounting is partial source-text characters/4, with unavailable conversation/tool/reasoning/cached usage explicitly unknown; these estimates do not establish total billed cost or an abstraction cost ranking. Shared infrastructure/reference correction/paper work are separate phases. KA-first and JACC reuse make this incremental production effort, not independent translations. No unsupported framework feature was observed for this selected configuration.

## Ontology explanation and paper follow-up
Use the existing operation/access classifications as grouping evidence, with implementation choices above recorded separately. Any runtime gap is an observation. Potential launch/configuration overhead, memory access behavior, compiler output and changed specialization are competing explanations, not demonstrated causes. No measured memory traffic, occupancy or roofline position was collected in this pass. Profiling and size scaling remain pending; this small configuration cannot support broad abstraction-selection recommendations. RQ1 is answered for this contract; RQ2 cost completeness is partial; RQ3 has matched measurements; RQ4 mechanisms and RQ5 generalization remain unresolved. Coordinator must export a portable snapshot and connect this observation to claims/figures/sections.
