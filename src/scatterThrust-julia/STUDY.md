# scatterThrust: verified four-model study observation

RQ IDs: RQ1–RQ5. Experiment IDs: E1–E5. Evidence ID: 20261017T000915846850Z_scatterThrust.
Configuration contract: `{"n": 4096, "repeat": 10, "types": ["Int8", "Int16", "Int32", "Int64", "Float32", "Float64"], "input": "T(i) andInt64reversepermutation n-1-i, originalfixture", "outputs": "all6*nscattervalues exactFloat64encoding", "timing": "sum6case means, perrepeatresetexcluded andperrepeatsynchronization retained"}`.
Protocol: five_trials_last_four_v1, all five correctness checks retained, timing mean trials2–5.
Evidence: `/home/imo/HeCBench_julia/.porting-state/portable_julia/runs/20261017T000915846850Z_scatterThrust/result.json`, config.json, timings.csv, raw logs, binary artifacts.
Ontology: ontology/benchmarks/scatterThrust.yaml; SHA256 `0bcffee33bbaac5b10fb706747673956ad76601e2aa48d48f80932dc196f05a6`. Original inferred ontology labels retained; no retroactive winner-based relabeling.

## Numerical confirmation
Each variant passes independent full-element CPU oracle checks and direct binary-array comparisons against native CUDA and CUDA.jl on every one of the five trials. Acceptance tolerances and all observed absolute/relative errors are preserved in result.json. NaN/nonfinite and shape/count checks are performed by the runner. Native device kernels are included unchanged; historical CUDA.jl original_main.jl and original_Makefile are retained. Parent compatibility entrypoints execute the historical original source. Study drivers record reference input alignment, any algorithm correction, and measurement handling as separate phases from the historical translation.

All six original data types are retained: Int8, Int16, Int32, Int64, Float32 and Float64. The original sourceT(i) and reverse Int64 permutation are shared exactly atn4096. Every one of24,576 outputs matches CPU and both GPU references exactly in all five trials. Native Thrust scatter is compared to authored CUDA.jl/KA/JACC device kernels; this is a vendor-primitive-to-kernel adaptation, not vendor algorithm equivalence. No collision or random-index scatter behavior is claimed. All study values are exactly representable in Float64 artifacts; arbitrary large Int64 values are not covered. Native signed narrow-integer conversion is checked on the recorded platform. Measurement preserves original per-internal-repeat reset exclusion and synchronization; the six per-case means are summed and retained separately. KA/JACC each succeed on candidate one, and JACC explicitly reserves zero shared memory. The historical Julia scatter algorithm already computes the actual benchmark.

## Runtime: microseconds
Five in-process trials are retained; the reported mean excludes trial1 and averages trials2–5. Each value uses the configuration's exact timed region and normalization. Notes above and `input_contract` distinguish per-repeat means from case or pipeline totals. Transfers, resets, allocation, temporary storage, graph control, host dispatch and synchronization are included where that named contract specifies. Compiler/JIT warmup, CPU correctness and artifact writing occur outside measurement. All GPU measurements use the shared GPU6 lock. Exact package/hardware evidence lives in setup and run bundles.

| Model | Trials1–5 (us) | Mean2–5 (us) |
|---|---|---|
| cuda | 68.7154, 68.0044, 71.1016, 70.0179, 70.0174 | 69.7853 |
| cuda_jl | 73.3328, 70.2889, 63.5271, 61.7687, 64.9913 | 65.1440 |
| ka | 98.0427, 92.5957, 84.6406, 80.7565, 77.8346 | 83.9569 |
| jacc | 131.4579, 115.4967, 103.8959, 107.1643, 105.1200 | 107.9192 |

## Attempts, tokens and feature support
Each new framework has one source candidate; zero compiler-feedback code repairs. Actual runner build/JIT invocations are recorded separately from timing trials; tool/reuse/setup failures are infrastructure events, not translation candidates. A five-trial timing series is not five translation attempts. Token accounting is partial source-text characters/4, with unavailable conversation/tool/reasoning/cached usage explicitly unknown; these estimates do not establish total billed cost or an abstraction cost ranking. Shared infrastructure/reference correction/paper work are separate phases. KA-first and JACC reuse make this incremental production effort, not independent translations. No unsupported framework feature was observed for this selected configuration.

## Ontology explanation and paper follow-up
Use the existing operation/access classifications as grouping evidence, with implementation choices above recorded separately. Any runtime gap is an observation. Potential launch/configuration overhead, memory access behavior, compiler output and changed specialization are competing explanations, not demonstrated causes. No measured memory traffic, occupancy or roofline position was collected in this pass. Profiling and size scaling remain pending; this small configuration cannot support broad abstraction-selection recommendations. RQ1 is answered for this contract; RQ2 cost completeness is partial; RQ3 has matched measurements; RQ4 mechanisms and RQ5 generalization remain unresolved. Coordinator must export a portable snapshot and connect this observation to claims/figures/sections.
