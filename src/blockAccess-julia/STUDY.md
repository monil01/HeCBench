# blockAccess: verified four-model study observation

RQ IDs: RQ1–RQ5. Experiment IDs: E1–E5. Evidence ID: 20261016T231543270781Z_blockAccess.
Configuration contract: `{"precision": "Float32toUInt8", "size": 65536, "internal_repeats": 10, "block_size": 256, "blocks": 64, "fixture": "inputzeroindexi=float((i*37)%1024-512)/4; negatives/fractional included", "cases": "reference,float4_blocked", "outputs": "2fullbytearrays convertedtoUInt32exactly", "timing": "sum2synchronizedloopmeans"}`.
Protocol: five_trials_last_four_v1, all five correctness checks retained, timing mean trials2–5.
Evidence: `/home/imo/HeCBench_julia/.porting-state/portable_julia/runs/20261016T231543270781Z_blockAccess/result.json`, config.json, timings.csv, raw logs, binary artifacts.
Ontology: ontology/benchmarks/blockAccess.yaml; SHA256 `61cb70ec3dda4c2a0fc9e1bb26218cee9e7a13f9e2b602ae367fb0b4cbcb08df`. Original inferred ontology labels retained; no retroactive winner-based relabeling.

## Numerical confirmation
Each variant passes independent full-element CPU oracle checks and direct binary-array comparisons against native CUDA and CUDA.jl on every one of the five trials. Acceptance tolerances and all observed absolute/relative errors are preserved in result.json. NaN/nonfinite and shape/count checks are performed by the runner. Native device kernels are included unchanged; historical CUDA.jl original_main.jl and original_Makefile are retained. Parent compatibility entrypoints execute the historical original source. Study drivers change only explicitly recorded reference inputs/measurement handling, a separate phase from original translation.

All2*65536float-to-byte outputs arecheckedexactly, includingnegative andfractionalvalues: Float32((zeroindex*37)%1024-512)/4, truncationtowardzero toInt32 thenmodulo256. NativeFloat4 reference andblockedCUB-styleBlockLoad/BlockStore4items retainunchangeddevicecode. HistoricalCUDA.jl scalargridstride conversion ispreserved; KA/JACCblockedcase maps4contiguousscalaritemsperthread withoutclaimingnativevectorinstructions. Namedgrid64blocks differsnative16*SMdefault. Referencehostmeasurementadapter initiallytriggeredJulia1.12LLVM LazyCallGraph hostcompilersegfault (R1), notGPU numericalfailure. Changinghostlaunchfunction uniondispatchtoexplicitnamedkernelbranches fixedit; nativeandKA5trialsets reusedunchanged. KAandJACC eachfirstdevicecandidate succeeds; referenceadapter has2candidates/2compilerruns and1hostcompilerrepair separatelyattributed. Performancegap mayreflectnativevectorization vsportable scalarloads aswellashostdispatch; noassemblycausalclaim.

## Runtime: microseconds
Each trial records the synchronized launch-loop mean per internal repeat. This includes host dispatch and excludes JIT/compilation, setup, allocations, transfers, input reset, CPU checking and artifact writing. Each trial starts with the same inputs. A device warmup precedes trial1. All timing commands are serialized on GPU6 via the shared lock. Native CUDA, CUDA.jl, KA and JACC are measured under the same named contract. Julia environments pin CUDA runtime12.0; exact package/hardware evidence lives in setup and run bundles.

| Model | Trials1–5 (us) | Mean2–5 (us) |
|---|---|---|
| cuda | 6.6560, 6.5090, 6.4380, 6.3591, 6.3681 | 6.4185 |
| cuda_jl | 9.0151, 10.7352, 8.8880, 9.2892, 9.0080 | 9.4801 |
| ka | 21.8522, 18.9382, 14.7082, 13.7572, 13.6411 | 15.2612 |
| jacc | 57.3186, 38.1543, 30.5223, 26.6212, 29.1153 | 31.1033 |

## Attempts, tokens and feature support
Each new framework has one source candidate; zero compiler-feedback code repairs. Actual runner build/JIT invocations are recorded separately from timing trials; tool/reuse/setup failures are infrastructure events, not translation candidates. A five-trial timing series is not five translation attempts. Token accounting is partial source-text characters/4, with unavailable conversation/tool/reasoning/cached usage explicitly unknown; these estimates do not establish total billed cost or an abstraction cost ranking. Shared infrastructure/reference correction/paper work are separate phases. KA-first and JACC reuse make this incremental production effort, not independent translations. No unsupported framework feature was observed for this selected configuration.

## Ontology explanation and paper follow-up
Use the existing operation/access classifications as grouping evidence, with implementation choices above recorded separately. Any runtime gap is an observation. Potential launch/configuration overhead, memory access behavior, compiler output and changed specialization are competing explanations, not demonstrated causes. No measured memory traffic, occupancy or roofline position was collected in this pass. Profiling and size scaling remain pending; this small configuration cannot support broad abstraction-selection recommendations. RQ1 is answered for this contract; RQ2 cost completeness is partial; RQ3 has matched measurements; RQ4 mechanisms and RQ5 generalization remain unresolved. Coordinator must export a portable snapshot and connect this observation to claims/figures/sections.
