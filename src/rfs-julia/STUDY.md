# rfs: verified four-model study observation

RQ IDs: RQ1–RQ5. Experiment IDs: E1–E5. Evidence ID: 20261016T232821091844Z_rfs.
Configuration contract: `{"precision": "Float32", "arrays": 16, "length": 512, "fixture": "raw=((i*1103515245+12345)&0xffffff)/2^24 signbyparity; firstelementeacharray1.0 =>actualmaxabs1", "rounding_factor": "createRoundingFactor(1,512)=1024 exactpower2", "cases": "16per-arrayatomiclaunches,1reverse-sumlaunch", "outputs": "32completearray-sumscalars, bitwiseCPUoracle", "timing": "sumphase1TOTAL16launches plusphase2singlelaunch (notphase1perarraynormalized)", "blocks_per_launch": 256, "threads_per_block": 256, "internal_repeats": 1}`.
Protocol: five_trials_last_four_v1, all five correctness checks retained, timing mean trials2–5.
Evidence: `/home/imo/HeCBench_julia/.porting-state/portable_julia/runs/20261016T232821091844Z_rfs/result.json`, config.json, timings.csv, raw logs, binary artifacts.
Ontology: ontology/benchmarks/rfs.yaml; SHA256 `2565050a03d774d4397e82c8069b58aa4984b84f01744ce8877012a8ff17f5de`. Original inferred ontology labels retained; no retroactive winner-based relabeling.

## Numerical confirmation
Each variant passes independent full-element CPU oracle checks and direct binary-array comparisons against native CUDA and CUDA.jl on every one of the five trials. Acceptance tolerances and all observed absolute/relative errors are preserved in result.json. NaN/nonfinite and shape/count checks are performed by the runner. Native device kernels are included unchanged; historical CUDA.jl original_main.jl and original_Makefile are retained. Parent compatibility entrypoints execute the historical original source. Study drivers change only explicitly recorded reference inputs/measurement handling, a separate phase from original translation.

Two original phases retained:16 array-sum launches plus reverse-array sum; first phase timing is total16 launches uniformly rather than printed native perarray mean. nArrays16,len512,r10 fixture maxabs1 forces factor1024 in all implementations. All32 rounded Float32 sums are bitwise equal to independent CPU oracle and directbothreferences in everytrial. Original native sumArrays pointer progression bug for arrays>65536 is outofscope; no untested sizeclaim. Shared fixture generatedwith historicalJulia deterministichash RNG, not originalC rand. Bothphase times retained separately.

## Runtime: microseconds
Each trial records the synchronized launch-loop mean per internal repeat. This includes host dispatch and excludes JIT/compilation, setup, allocations, transfers, input reset, CPU checking and artifact writing. Each trial starts with the same inputs. A device warmup precedes trial1. All timing commands are serialized on GPU6 via the shared lock. Native CUDA, CUDA.jl, KA and JACC are measured under the same named contract. Julia environments pin CUDA runtime12.0; exact package/hardware evidence lives in setup and run bundles.

| Model | Trials1–5 (us) | Mean2–5 (us) |
|---|---|---|
| cuda | 84.7500, 86.0600, 86.9810, 87.0700, 87.7000 | 86.9528 |
| cuda_jl | 167.9210, 190.7820, 143.2700, 140.0800, 137.7920 | 152.9810 |
| ka | 174.8120, 194.1110, 138.8610, 134.6410, 133.0210 | 150.1585 |
| jacc | 234.3120, 358.5340, 198.2520, 186.1320, 185.4210 | 232.0848 |

## Attempts, tokens and feature support
Each new framework has one source candidate; zero compiler-feedback code repairs. Actual runner build/JIT invocations are recorded separately from timing trials; tool/reuse/setup failures are infrastructure events, not translation candidates. A five-trial timing series is not five translation attempts. Token accounting is partial source-text characters/4, with unavailable conversation/tool/reasoning/cached usage explicitly unknown; these estimates do not establish total billed cost or an abstraction cost ranking. Shared infrastructure/reference correction/paper work are separate phases. KA-first and JACC reuse make this incremental production effort, not independent translations. No unsupported framework feature was observed for this selected configuration.

## Ontology explanation and paper follow-up
Use the existing operation/access classifications as grouping evidence, with implementation choices above recorded separately. Any runtime gap is an observation. Potential launch/configuration overhead, memory access behavior, compiler output and changed specialization are competing explanations, not demonstrated causes. No measured memory traffic, occupancy or roofline position was collected in this pass. Profiling and size scaling remain pending; this small configuration cannot support broad abstraction-selection recommendations. RQ1 is answered for this contract; RQ2 cost completeness is partial; RQ3 has matched measurements; RQ4 mechanisms and RQ5 generalization remain unresolved. Coordinator must export a portable snapshot and connect this observation to claims/figures/sections.
