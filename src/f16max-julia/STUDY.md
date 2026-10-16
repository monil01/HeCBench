# f16max: verified four-model study observation

RQ IDs: RQ1–RQ5. Experiment IDs: E1–E5. Evidence ID: 20261016T234953198276Z_f16max.
Configuration contract: `{"n_half2": 4096, "scalar_half_count": 8192, "repeat": 10, "fixture": "half((i17)%257-128)+.125 andhalf((i31)%251-125)+.375; finitevaryingsignedvalues", "outputs": "Bothpacked/scalar8192halfoutputs encodedFloat64exact", "timing": "sum2case synchronizedloopmeans; setup/copiesexcluded"}`.
Protocol: five_trials_last_four_v1, all five correctness checks retained, timing mean trials2–5.
Evidence: `/home/imo/HeCBench_julia/.porting-state/portable_julia/runs/20261016T234953198276Z_f16max/result.json`, config.json, timings.csv, raw logs, binary artifacts.
Ontology: ontology/benchmarks/f16max.yaml; SHA256 `da445ec6c866c24f738c91f6dc7fff16e15e149f516169d814c6cbf8b09307ad`. Original inferred ontology labels retained; no retroactive winner-based relabeling.

## Numerical confirmation
Each variant passes independent full-element CPU oracle checks and direct binary-array comparisons against native CUDA and CUDA.jl on every one of the five trials. Acceptance tolerances and all observed absolute/relative errors are preserved in result.json. NaN/nonfinite and shape/count checks are performed by the runner. Native device kernels are included unchanged; historical CUDA.jl original_main.jl and original_Makefile are retained. Parent compatibility entrypoints execute the historical original source. Study drivers change only explicitly recorded reference inputs/measurement handling, a separate phase from original translation.

Bothnativepackedhalf2 andscalarhalf paths retained, each8192Float16outputvalues exportedFloat64exact,16384total. nHalf24096,r10 namedfixture usesvaryingsignedfinitehalfinputswithnoNaNs/Inf/signedzero/ties. Originalrandup922021 overflowsbinary16; originalfabs(NaN)>tol check canfailtodetectNaNs, sonofalseoriginal-inputcorrectnessclaim. Nativehalfsubsign+bytepermutation versusJulia/KA/JACC maxsemanticmapping: packedpath2lanesperthread, scalar1lane, notrepresentation-intrinsic-equivalent. HistoricalJuliascalarpathduplicatedtwice referencecorrectedtopacked2itemsvsscalar1itemmapping separately. ExactCPUmax andbothrefsall5. Originalpackedatol1e-2/scalar1e-3 strengthenedtozero forthisexactfinitefixture, neverloosened. Allportablekernels candidate1, noJACCsharedallocation.

## Runtime: microseconds
Each trial records the synchronized launch-loop mean per internal repeat. This includes host dispatch and excludes JIT/compilation, setup, allocations, transfers, input reset, CPU checking and artifact writing. Each trial starts with the same inputs. A device warmup precedes trial1. All timing commands are serialized on GPU6 via the shared lock. Native CUDA, CUDA.jl, KA and JACC are measured under the same named contract. Julia environments pin CUDA runtime12.0; exact package/hardware evidence lives in setup and run bundles.

| Model | Trials1–5 (us) | Mean2–5 (us) |
|---|---|---|
| cuda | 5.9361, 7.3111, 5.9590, 5.8540, 5.8730 | 6.2493 |
| cuda_jl | 9.5850, 12.4580, 8.6990, 8.7941, 8.1040 | 9.5138 |
| ka | 18.7431, 22.1361, 15.9630, 15.8780, 14.6261 | 17.1508 |
| jacc | 22.6012, 26.1762, 19.0432, 19.0532, 18.0182 | 20.5727 |

## Attempts, tokens and feature support
Each new framework has one source candidate; zero compiler-feedback code repairs. Actual runner build/JIT invocations are recorded separately from timing trials; tool/reuse/setup failures are infrastructure events, not translation candidates. A five-trial timing series is not five translation attempts. Token accounting is partial source-text characters/4, with unavailable conversation/tool/reasoning/cached usage explicitly unknown; these estimates do not establish total billed cost or an abstraction cost ranking. Shared infrastructure/reference correction/paper work are separate phases. KA-first and JACC reuse make this incremental production effort, not independent translations. No unsupported framework feature was observed for this selected configuration.

## Ontology explanation and paper follow-up
Use the existing operation/access classifications as grouping evidence, with implementation choices above recorded separately. Any runtime gap is an observation. Potential launch/configuration overhead, memory access behavior, compiler output and changed specialization are competing explanations, not demonstrated causes. No measured memory traffic, occupancy or roofline position was collected in this pass. Profiling and size scaling remain pending; this small configuration cannot support broad abstraction-selection recommendations. RQ1 is answered for this contract; RQ2 cost completeness is partial; RQ3 has matched measurements; RQ4 mechanisms and RQ5 generalization remain unresolved. Coordinator must export a portable snapshot and connect this observation to claims/figures/sections.
