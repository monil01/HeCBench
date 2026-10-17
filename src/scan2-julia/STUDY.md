# scan2: verified four-model study observation

RQ IDs: RQ1–RQ5. Experiment IDs: E1–E5. Evidence ID: 20261017T001239341253Z_scan2.
Configuration contract: `{"n": 4096, "repeat": 10, "block_elements": 256, "hierarchy_passes": 1, "stage_launches": 3, "fixture": "Float32((i17)%13+1), nonzeroexactintegers", "outputs": "all4096globalexclusiveprefixvalues exactFloat64encoding", "timing": "complete3stagepipeline synchronizedloopmean"}`.
Protocol: five_trials_last_four_v1, all five correctness checks retained, timing mean trials2–5.
Evidence: `/home/imo/HeCBench_julia/.porting-state/portable_julia/runs/20261017T001239341253Z_scan2/result.json`, config.json, timings.csv, raw logs, binary artifacts.
Ontology: ontology/benchmarks/scan2.yaml; SHA256 `79bdb4e5a3fd75dd2d66b8d5c48fdfb9b9aef4399bb649020cb1ee1cc200a514`. Original inferred ontology labels retained; no retroactive winner-based relabeling.

## Numerical confirmation
Each variant passes independent full-element CPU oracle checks and direct binary-array comparisons against native CUDA and CUDA.jl on every one of the five trials. Acceptance tolerances and all observed absolute/relative errors are preserved in result.json. NaN/nonfinite and shape/count checks are performed by the runner. Native device kernels are included unchanged; historical CUDA.jl original_main.jl and original_Makefile are retained. Parent compatibility entrypoints execute the historical original source. Study drivers record reference input alignment, any algorithm correction, and measurement handling as separate phases from the historical translation.

The native three-stage global scan is preserved for named n4096/block256:16 parallel block scans, a16-element top scan, and16 block-addition workgroups. All4096 nonzero-input prefix values match CPU and both references exactly on every trial. Historical single-GPU-thread serial scan is separately corrected to the native parallel algorithm. The top prefix retains the native extra idle strides through block_size256. Native redundantly writes identical block sums from every thread; the study Julia kernels restrict that global write to thread zero. The named integer-valued Float32 fixture avoids association drift and strengthens the original relativeL2.001 acceptance to exact outputs. Only one hierarchy pass is covered; larger recursive hierarchies are untested. JACC explicitly reserves1024/64/4 shared bytes and copies zero workspaces for its shared arrays. Both frameworks succeed on candidate one, with three actual stage specializations per process. Runtime covers the entire three-launch pipeline and final synchronization, excluding buffer setup and output copies.

## Runtime: microseconds
Five in-process trials are retained; the reported mean excludes trial1 and averages trials2–5. Each value uses the configuration's exact timed region and normalization. Notes above and `input_contract` distinguish per-repeat means from case or pipeline totals. Transfers, resets, allocation, temporary storage, graph control, host dispatch and synchronization are included where that named contract specifies. Compiler/JIT warmup, CPU correctness and artifact writing occur outside measurement. All GPU measurements use the shared GPU6 lock. Exact package/hardware evidence lives in setup and run bundles.

| Model | Trials1–5 (us) | Mean2–5 (us) |
|---|---|---|
| cuda | 9.0521, 9.1891, 9.1481, 8.9941, 8.9911 | 9.0806 |
| cuda_jl | 11.3961, 19.6171, 11.4731, 11.6581, 9.9091 | 13.1643 |
| ka | 27.9672, 26.4632, 17.8711, 19.1292, 17.4742 | 20.2344 |
| jacc | 32.7953, 36.7683, 26.9912, 24.8523, 25.4152 | 28.5068 |

## Attempts, tokens and feature support
Each new framework has one source candidate; zero compiler-feedback code repairs. Actual runner build/JIT invocations are recorded separately from timing trials; tool/reuse/setup failures are infrastructure events, not translation candidates. A five-trial timing series is not five translation attempts. Token accounting is partial source-text characters/4, with unavailable conversation/tool/reasoning/cached usage explicitly unknown; these estimates do not establish total billed cost or an abstraction cost ranking. Shared infrastructure/reference correction/paper work are separate phases. KA-first and JACC reuse make this incremental production effort, not independent translations. No unsupported framework feature was observed for this selected configuration.

## Ontology explanation and paper follow-up
Use the existing operation/access classifications as grouping evidence, with implementation choices above recorded separately. Any runtime gap is an observation. Potential launch/configuration overhead, memory access behavior, compiler output and changed specialization are competing explanations, not demonstrated causes. No measured memory traffic, occupancy or roofline position was collected in this pass. Profiling and size scaling remain pending; this small configuration cannot support broad abstraction-selection recommendations. RQ1 is answered for this contract; RQ2 cost completeness is partial; RQ3 has matched measurements; RQ4 mechanisms and RQ5 generalization remain unresolved. Coordinator must export a portable snapshot and connect this observation to claims/figures/sections.
