# atomicCAS: verified four-model study observation

RQ IDs: RQ1–RQ5. Experiment IDs: E1–E5. Evidence ID: 20261016T225232272322Z_atomicCAS.
Configuration contract: `{"precision": "UInt64,Int64,Float64 rawbitCAS", "threads": 4096, "block_size": 256, "cases": "min3types,max3types,add3types", "minmax_repeats": 10, "add_repeats": 1, "fixture": "Native initialmaxforMin; unsigned0/signedlowest/floatminforMax; zeroAdd", "timing": "sum9 separatelysynchronizedloopmeans; individualcase timesretained", "outputs": "9 numericfinalvalues convertedexactly toFloat64 atn4096"}`.
Protocol: five_trials_last_four_v1, all five correctness checks retained, timing mean trials2–5.
Evidence: `/home/imo/HeCBench_julia/.porting-state/portable_julia/runs/20261016T225232272322Z_atomicCAS/result.json`, config.json, timings.csv, raw logs, binary artifacts.
Ontology: ontology/benchmarks/atomicCAS.yaml; SHA256 `fe3827c8ef91e9c1dfeb88d47b5a3578b5b273169186dd4ad7d9f684808f0897`. Original inferred ontology labels retained; no retroactive winner-based relabeling.

## Numerical confirmation
Each variant passes independent full-element CPU oracle checks and direct binary-array comparisons against native CUDA and CUDA.jl on every one of the five trials. Acceptance tolerances and all observed absolute/relative errors are preserved in result.json. NaN/nonfinite and shape/count checks are performed by the runner. Native device kernels are included unchanged; historical CUDA.jl original_main.jl and original_Makefile are retained. Parent compatibility entrypoints execute the historical original source. Study drivers change only explicitly recorded reference inputs/measurement handling, a separate phase from original translation.

Historical CUDA.jl was a semantic stub: a singlethread wrote expectedmin/max/sum results and performed noatomicoperations. The study reference is a separately attributed correctness repair using genuine perthread compare-and-swap loops overUInt64 rawbits, for min/max/add inUInt64,Int64,Float64; original source andtokens preserved. KA andJACC use Atomix.@atomicreplace with identical loops, notnativeatomicadd substitution. Namedstudy uses4096threads versusnativedefault262144; min/max10internalrepeats, add1asnative. Runtime sums9case means; everycase time retained. Eachof9completeoutputscalars matchesoracle andbothGPUrefs all5. NoNaNinputs tested. KA has TWOsourcecandidates and ONEcompiler-feedbackrepair: @kernel forbidsreturn, terminalreturn removed (B2). JACCcandidate1succeeds. Referencecorrection candidate1 independently succeeds. CASmechanism/contendedthreadparticipation ratherthanfinalvalue-only fidelity isconfirmed bysourceinspection; noassemblyprofilingclaim.

## Runtime: microseconds
Each trial records the synchronized launch-loop mean per internal repeat. This includes host dispatch and excludes JIT/compilation, setup, allocations, transfers, input reset, CPU checking and artifact writing. Each trial starts with the same inputs. A device warmup precedes trial1. All timing commands are serialized on GPU6 via the shared lock. Native CUDA, CUDA.jl, KA and JACC are measured under the same named contract. Julia environments pin CUDA runtime12.0; exact package/hardware evidence lives in setup and run bundles.

| Model | Trials1–5 (us) | Mean2–5 (us) |
|---|---|---|
| cuda | 10191.4058, 10212.6049, 10195.5059, 10209.6970, 10246.1142 | 10215.9805 |
| cuda_jl | 146035.8761, 137516.2646, 145957.7497, 144654.2529, 145114.1447 | 143310.6030 |
| ka | 84080.5489, 81360.1286, 79540.9893, 82578.7409, 81951.6558 | 81357.8787 |
| jacc | 78976.5353, 77709.2953, 79089.1591, 77327.2181, 78124.8542 | 78062.6317 |

## Attempts, tokens and feature support
KA has two source candidates and one compiler-feedback repair (B2 terminal return forbidden); JACC has one source candidate and zero compiler-feedback repairs. The separate CUDA.jl correctness repair has one candidate and one compiler check. Actual runner build/JIT invocations are recorded separately from timing trials; tool/reuse/setup failures are infrastructure events, not translation candidates. A five-trial timing series is not five translation attempts. Token accounting is partial source-text characters/4, with unavailable conversation/tool/reasoning/cached usage explicitly unknown; these estimates do not establish total billed cost or an abstraction cost ranking. Shared infrastructure/reference correction/paper work are separate phases. KA-first and JACC reuse make this incremental production effort, not independent translations. No unsupported framework feature was observed for this selected configuration.

## Ontology explanation and paper follow-up
Use the existing operation/access classifications as grouping evidence, with implementation choices above recorded separately. Any runtime gap is an observation. Potential launch/configuration overhead, memory access behavior, compiler output and changed specialization are competing explanations, not demonstrated causes. No measured memory traffic, occupancy or roofline position was collected in this pass. Profiling and size scaling remain pending; this small configuration cannot support broad abstraction-selection recommendations. RQ1 is answered for this contract; RQ2 cost completeness is partial; RQ3 has matched measurements; RQ4 mechanisms and RQ5 generalization remain unresolved. Coordinator must export a portable snapshot and connect this observation to claims/figures/sections.
