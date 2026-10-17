# wedford: verified four-model study observation

RQ IDs: RQ1–RQ5. Experiment IDs: E1–E5. Evidence ID: 20261017T003003745416Z_wedford.
Configuration contract: `{"n": 8192, "repeat": 10, "batch": 16, "spatial": 64, "features": 8, "threads_mapping": "native32x16 ->portablelinear512, identicaltx/tyworkassignment", "input": "Float32((i17)%1024)/1024 commonfixture", "outputs": "8means+8populationvariances Float64exactencoding", "oracle": "independentFloat64twopassmean/deviationvariance, nativeatol1e-3unchanged", "timing": "synchronizedcompleteWelfordkernelmean"}`.
Protocol: five_trials_last_four_v1, all five correctness checks retained, timing mean trials2–5.
Evidence: `/home/imo/HeCBench_julia/.porting-state/portable_julia/runs/20261017T003003745416Z_wedford/result.json`, config.json, timings.csv, raw logs, binary artifacts.
Ontology: ontology/benchmarks/wedford.yaml; SHA256 `be0492c9a9be6c4c7514f1676d81f753b38ad821e0fc636a2596fc45aa6c313b`. Original inferred ontology labels retained; no retroactive winner-based relabeling.

## Numerical confirmation
Each variant passes independent full-element CPU oracle checks and direct binary-array comparisons against native CUDA and CUDA.jl on every one of the five trials. Acceptance tolerances and all observed absolute/relative errors are preserved in result.json. NaN/nonfinite and shape/count checks are performed by the runner. Native device kernels are included unchanged; historical CUDA.jl original_main.jl and original_Makefile are retained. Parent compatibility entrypoints execute the historical original source. Study drivers record reference input alignment, any algorithm correction, and measurement handling as separate phases from the historical translation.

The native Float32 sequential Welford update and counted mean/M2 merge are implemented, replacing the historical Julia sum/sumsquares approximation in a separate reference-correction phase. Native32x16 threads map to a512-thread linear workgroup with identical batch/spatial assignment. Warp reductions are adapted to shared-memory emulation with the same merge tree and explicit full-block barriers; native uses shuffles and warp synchronization. Counts remain Int32, stored bitwise alongside Float32 mean/M2 in aUInt32 shared workspace. Portable shared use is6144 bytes versus native640 bytes, and JACC additionally copies the zero workspace; these are measured-source resource differences, not a demonstrated performance bottleneck. All16 mean/population-variance outputs match both GPU references exactly across five trials and satisfy an independent Float64 two-pass mean/deviation oracle at the unchanged native atol1e-3. The shared dyadic input fixture replaces incompatible initialization, with batch16/spatial64/features8,r10. Other shapes and high-offset numerical stability are untested. Both frameworks succeed on candidate one. This is an explicit native-shuffle-to-shared algorithm adaptation using framework device kernels.

## Runtime: microseconds
Five in-process trials are retained; the mean excludes trial1 and averages trials2–5. Values measure the complete synchronized Welford kernel loop mean specified in input_contract. Setup, transfer, CPU validation and artifact writing are excluded. Compiler/JIT warmup precedes allfive trials; all GPU commands use the GPU6 shared lock.

| Model | Trials1–5 (us) | Mean2–5 (us) |
|---|---|---|
| cuda | 5.7711, 6.0861, 5.9641, 5.8521, 5.8451 | 5.9369 |
| cuda_jl | 9.5391, 11.0061, 8.5781, 8.3821, 8.3591 | 9.0814 |
| ka | 24.1202, 15.4782, 10.3320, 9.8661, 9.5441 | 11.3051 |
| jacc | 34.7613, 25.8052, 15.3531, 14.0361, 13.6642 | 17.2147 |

## Attempts, tokens and feature support
Each new framework has one source candidate; zero compiler-feedback code repairs. Actual runner build/JIT invocations are recorded separately from timing trials; tool/reuse/setup failures are infrastructure events, not translation candidates. A five-trial timing series is not five translation attempts. Token accounting is partial source-text characters/4, with unavailable conversation/tool/reasoning/cached usage explicitly unknown; these estimates do not establish total billed cost or an abstraction cost ranking. Shared infrastructure/reference correction/paper work are separate phases. KA-first and JACC reuse make this incremental production effort, not independent translations. No unsupported framework feature was observed for this selected configuration.

## Ontology explanation and paper follow-up
Use the existing operation/access classifications as grouping evidence, with implementation choices above recorded separately. Any runtime gap is an observation. Potential launch/configuration overhead, memory access behavior, compiler output and changed specialization are competing explanations, not demonstrated causes. No measured memory traffic, occupancy or roofline position was collected in this pass. Profiling and size scaling remain pending; this small configuration cannot support broad abstraction-selection recommendations. RQ1 is answered for this contract; RQ2 cost completeness is partial; RQ3 has matched measurements; RQ4 mechanisms and RQ5 generalization remain unresolved. Coordinator must export a portable snapshot and connect this observation to claims/figures/sections.
