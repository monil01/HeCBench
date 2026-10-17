# unfold: verified four-model study observation

RQ IDs: RQ1–RQ5. Experiment IDs: E1–E5. Evidence ID: 20261016T224309012042Z_unfold.
Configuration contract: `{"precision": "int32", "size": 4096, "internal_repeats": 100, "fixture": "input[i]=zero_based_i%256; idxdim0,size2,step1,strides1; exact output100*input"}`.
Protocol: five_trials_last_four_v1, all five correctness checks retained, timing mean trials2–5.
Evidence: `/home/imo/HeCBench_julia/.porting-state/portable_julia/runs/20261016T224309012042Z_unfold/result.json`, config.json, timings.csv, raw logs, binary artifacts.
Ontology: ontology/benchmarks/unfold.yaml; SHA256 `cc44f4a3ad8c5c6a6a9ce1a646da2bc147b1f4d96f47fe2a57a0941e73661e51`. Original inferred ontology labels retained; no retroactive winner-based relabeling.

## Numerical confirmation
Each variant passes independent full-element CPU oracle checks and direct binary-array comparisons against native CUDA and CUDA.jl on every one of the five trials. Acceptance tolerances and all observed absolute/relative errors are preserved in result.json. NaN/nonfinite and shape/count checks are performed by the runner. Native device kernels are included unchanged; historical CUDA.jl original_main.jl and original_Makefile are retained. Parent compatibility entrypoints execute the historical original source. Study drivers record reference input alignment, any algorithm correction, and measurement handling as separate phases from the historical translation.

The native fixed idx_dim0,size2,step1,strides1 configuration has exactly one input contribution per output; ports preserve this fixed benchmark configuration, not arbitrary unfold settings. Shared deterministic input i%256 replaces distinct native rand and historical Julia RNG streams. Native launch uses64threads and4elements per thread; existing CUDA.jl and new variants map one element per thread with256thread workgroups. This mapping difference is a performance confound and source-level observation. Integer outputs checked exactly4096values allfive trials.

## Runtime: microseconds
Five in-process trials are retained; the reported mean excludes trial1 and averages trials2–5. Each value uses the configuration's exact timed region and normalization. Notes above and `input_contract` distinguish per-repeat means from case or pipeline totals. Transfers, resets, allocation, temporary storage, graph control, host dispatch and synchronization are included where that named contract specifies. Compiler/JIT warmup, CPU correctness and artifact writing occur outside measurement. All GPU measurements use the shared GPU6 lock. Exact package/hardware evidence lives in setup and run bundles.

| Model | Trials1–5 (us) | Mean2–5 (us) |
|---|---|---|
| cuda | 4.8068, 4.7904, 4.8035, 4.8129, 4.7999 | 4.8017 |
| cuda_jl | 2.5097, 2.6348, 2.5905, 2.4577, 2.5399 | 2.5557 |
| ka | 4.6839, 5.2968, 4.5905, 4.1719, 4.2314 | 4.5727 |
| jacc | 7.2061, 6.9853, 6.4377, 6.2388, 6.2900 | 6.4879 |

## Attempts, tokens and feature support
Each new framework has one source candidate; zero compiler-feedback code repairs. Actual runner build/JIT invocations are recorded separately from timing trials; tool/reuse/setup failures are infrastructure events, not translation candidates. A five-trial timing series is not five translation attempts. Token accounting is partial source-text characters/4, with unavailable conversation/tool/reasoning/cached usage explicitly unknown; these estimates do not establish total billed cost or an abstraction cost ranking. Shared infrastructure/reference correction/paper work are separate phases. KA-first and JACC reuse make this incremental production effort, not independent translations. No unsupported framework feature was observed for this selected configuration.

## Ontology explanation and paper follow-up
Use the existing operation/access classifications as grouping evidence, with implementation choices above recorded separately. Any runtime gap is an observation. Potential launch/configuration overhead, memory access behavior, compiler output and changed specialization are competing explanations, not demonstrated causes. No measured memory traffic, occupancy or roofline position was collected in this pass. Profiling and size scaling remain pending; this small configuration cannot support broad abstraction-selection recommendations. RQ1 is answered for this contract; RQ2 cost completeness is partial; RQ3 has matched measurements; RQ4 mechanisms and RQ5 generalization remain unresolved. Coordinator must export a portable snapshot and connect this observation to claims/figures/sections.
