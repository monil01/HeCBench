# scan: verified four-model study observation

RQ IDs: RQ1–RQ5. Experiment IDs: E1–E5. Evidence ID: 20261017T001210760988Z_scan.
Configuration contract: `{"n": 4096, "repeat": 3, "block_elements": [128, 256, 512, 1024, 2048], "types": ["Int8", "Int16", "Int32", "Int64"], "modes": ["unaware", "bank_conflict_aware"], "grid_blocks": 1280, "fixture": "T((i17)%5+1), samepositive1..5rangeasnative; everyblockindependentexclusiveprefix", "outputs": "All40*nexactresults Float64encoding", "timing": "sum40kernel synchronizedloopmeans; setup/copyexcluded"}`.
Protocol: five_trials_last_four_v1, all five correctness checks retained, timing mean trials2–5.
Evidence: `/home/imo/HeCBench_julia/.porting-state/portable_julia/runs/20261017T001210760988Z_scan/result.json`, config.json, timings.csv, raw logs, binary artifacts.
Ontology: ontology/benchmarks/scan.yaml; SHA256 `1fd18900cb492c73e4f3af6d977542b74aac9aa38947e29add7dea27d5c0995a`. Original inferred ontology labels retained; no retroactive winner-based relabeling.

## Numerical confirmation
Each variant passes independent full-element CPU oracle checks and direct binary-array comparisons against native CUDA and CUDA.jl on every one of the five trials. Acceptance tolerances and all observed absolute/relative errors are preserved in result.json. NaN/nonfinite and shape/count checks are performed by the runner. Native device kernels are included unchanged; historical CUDA.jl original_main.jl and original_Makefile are retained. Parent compatibility entrypoints execute the historical original source. Study drivers change only explicitly recorded reference inputs/measurement handling, a separate phase from original translation.

All40 native configurations are covered: block lengths128/256/512/1024/2048, signed8/16/32/64-bit storage, and unpadded versus bank-conflict-aware Blelloch scans. Historical one-thread serial scans with duplicate labels are separately corrected to the actual native upsweep/downsweep and padded indexing. Native grid1280 (=16*80SM) and N/2 threads are retained in every model; n4096 means many blocks are inactive. All163,840 block-local exclusive-prefix outputs match CPU and both references exactly in every trial. Int8 accumulation wraps consistently with native char on this platform. JACC candidate one hit a real register-limit launch failure during N1024/Int8 warmup:136 registers*512 threads=69,632 exceeds65,536. Public LaunchSpec has no maxregs field. Candidate two uses bounded Int32 indices and safe @inbounds, preserving the algorithm, geometry and all40 cases, and passes. This is an R1 resource-launch repair, not an unsupported feature. JACC copies zero workspaces only in active blocks, with exact N or2N storage bytes; this copy is absent in native/KA and remains a confound. Forty final kernel specializations are not forty translation attempts.

## Runtime: microseconds
Each trial records the synchronized launch-loop mean per internal repeat. This includes host dispatch and excludes JIT/compilation, setup, allocations, transfers, input reset, CPU checking and artifact writing. Each trial starts with the same inputs. A device warmup precedes trial1. All timing commands are serialized on GPU6 via the shared lock. Native CUDA, CUDA.jl, KA and JACC are measured under the same named contract. Julia environments pin CUDA runtime12.0; exact package/hardware evidence lives in setup and run bundles.

| Model | Trials1–5 (us) | Mean2–5 (us) |
|---|---|---|
| cuda | 303.4360, 310.6120, 297.0110, 299.9057, 297.4367 | 301.2413 |
| cuda_jl | 499.9487, 559.9560, 451.1290, 409.7470, 397.7173 | 454.6373 |
| ka | 703.7893, 845.7387, 562.3203, 540.0570, 601.8683 | 637.4961 |
| jacc | 12621.9007, 3033.8533, 2849.8523, 2775.4647, 2743.0870 | 2850.5643 |

## Attempts, tokens and feature support
KA has one candidate/one process check. JACC has two candidates/two process checks, zero kernel compiler-feedback repairs and one R1 launch-resource repair. Actual runner build/JIT invocations are recorded separately from timing trials; tool/reuse/setup failures are infrastructure events, not translation candidates. A five-trial timing series is not five translation attempts. Token accounting is partial source-text characters/4, with unavailable conversation/tool/reasoning/cached usage explicitly unknown; these estimates do not establish total billed cost or an abstraction cost ranking. Shared infrastructure/reference correction/paper work are separate phases. KA-first and JACC reuse make this incremental production effort, not independent translations. No unsupported framework feature was observed for this selected configuration.

## Ontology explanation and paper follow-up
Use the existing operation/access classifications as grouping evidence, with implementation choices above recorded separately. Any runtime gap is an observation. Potential launch/configuration overhead, memory access behavior, compiler output and changed specialization are competing explanations, not demonstrated causes. No measured memory traffic, occupancy or roofline position was collected in this pass. Profiling and size scaling remain pending; this small configuration cannot support broad abstraction-selection recommendations. RQ1 is answered for this contract; RQ2 cost completeness is partial; RQ3 has matched measurements; RQ4 mechanisms and RQ5 generalization remain unresolved. Coordinator must export a portable snapshot and connect this observation to claims/figures/sections.
