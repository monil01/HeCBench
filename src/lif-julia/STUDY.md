# lif: verified four-model study observation

RQ IDs: RQ1–RQ5. Experiment IDs: E1–E5. Evidence ID: 20261016T233624552647Z_lif.
Configuration contract: `{"precision": "Float32", "neurons_per_item": 64, "items": 64, "steps": 10, "fixture": "Originalglibc srand123/rand sequence andinitializationorder identicalJulia ccallandC++; dt.1,tau_rc10,tau_ref2", "outputs": "All4096voltage,4096refractorytimes,4096spikes concatenated", "timing": "meanperstep synchronizedloop, setup/reset/transfers excluded", "acceptance": "atol1e-4+rtol1e-5*abs(reference) forall3statearrays; finitechecks", "floating_point": "Native --fmad=false aligns noncontracted CPU/Julia expressions; original unmodified default native build failed strengthened refractory state tolerance max_abs0.0394287"}`.
Protocol: five_trials_last_four_v1, all five correctness checks retained, timing mean trials2–5.
Evidence: `/home/imo/HeCBench_julia/.porting-state/portable_julia/runs/20261016T233624552647Z_lif/result.json`, config.json, timings.csv, raw logs, binary artifacts.
Ontology: ontology/benchmarks/lif.yaml; SHA256 `cc3fc2ac2897da57d08f0c8fc216ffeea419140e753e163333cb23d1afd5d5cf`. Original inferred ontology labels retained; no retroactive winner-based relabeling.

## Numerical confirmation
Each variant passes independent full-element CPU oracle checks and direct binary-array comparisons against native CUDA and CUDA.jl on every one of the five trials. Acceptance tolerances and all observed absolute/relative errors are preserved in result.json. NaN/nonfinite and shape/count checks are performed by the runner. Native device kernels are included unchanged; historical CUDA.jl original_main.jl and original_Makefile are retained. Parent compatibility entrypoints execute the historical original source. Study drivers change only explicitly recorded reference inputs/measurement handling, a separate phase from original translation.

Native default FMAcontraction FAILED strengthened full-stateCPUcheck max_abs0.0394287109 (bundle233011899244Z), whileJulia/KA matched. Namedcontract noncontracted_v2 explicitly compiles unchanged nativekernel with --fmad=false, yielding all12288statevalues EXACTacrossCPU/CUDA/CUDAjl/KA/JACC everytrial. No tolerancewaswidened. This compiler arithmeticchoice is a confound and corrective reference/measurementphase, not a KAcompilerrepair. Original filteredspike-time tolerance0.1 and mincount check couldbevacuous; study strengthensfullvoltage/refractory/spike state atol1e-4 rtol1e-5. Selected64neuronsx64items10steps, sharedoriginalglibcrand123 initialization requiresplatformprovenance. CUDAjl/KA source candidate1 and JACCcandidate1; nativeadapter configuration2.

## Runtime: microseconds
Each trial records the synchronized launch-loop mean per internal repeat. This includes host dispatch and excludes JIT/compilation, setup, allocations, transfers, input reset, CPU checking and artifact writing. Each trial starts with the same inputs. A device warmup precedes trial1. All timing commands are serialized on GPU6 via the shared lock. Native CUDA, CUDA.jl, KA and JACC are measured under the same named contract. Julia environments pin CUDA runtime12.0; exact package/hardware evidence lives in setup and run bundles.

| Model | Trials1–5 (us) | Mean2–5 (us) |
|---|---|---|
| cuda | 3.5850, 3.4130, 3.3960, 3.4150, 3.3801 | 3.4010 |
| cuda_jl | 7.2671, 8.5991, 5.9710, 6.3031, 5.7651 | 6.6596 |
| ka | 11.9801, 12.7822, 11.2091, 8.9881, 8.9561 | 10.4839 |
| jacc | 23.2292, 22.8572, 14.6762, 16.1751, 15.4652 | 17.2934 |

## Attempts, tokens and feature support
Each new framework has one source candidate; zero compiler-feedback code repairs. Actual runner build/JIT invocations are recorded separately from timing trials; tool/reuse/setup failures are infrastructure events, not translation candidates. A five-trial timing series is not five translation attempts. Token accounting is partial source-text characters/4, with unavailable conversation/tool/reasoning/cached usage explicitly unknown; these estimates do not establish total billed cost or an abstraction cost ranking. Shared infrastructure/reference correction/paper work are separate phases. KA-first and JACC reuse make this incremental production effort, not independent translations. No unsupported framework feature was observed for this selected configuration.

## Ontology explanation and paper follow-up
Use the existing operation/access classifications as grouping evidence, with implementation choices above recorded separately. Any runtime gap is an observation. Potential launch/configuration overhead, memory access behavior, compiler output and changed specialization are competing explanations, not demonstrated causes. No measured memory traffic, occupancy or roofline position was collected in this pass. Profiling and size scaling remain pending; this small configuration cannot support broad abstraction-selection recommendations. RQ1 is answered for this contract; RQ2 cost completeness is partial; RQ3 has matched measurements; RQ4 mechanisms and RQ5 generalization remain unresolved. Coordinator must export a portable snapshot and connect this observation to claims/figures/sections.

Accounting clarification: KA has one source candidate and two process/JIT checks because the native compiler configuration correction required a matched rerun; JACC has one candidate and one process/JIT check. The additional KA process is not a compiler-feedback source repair.
