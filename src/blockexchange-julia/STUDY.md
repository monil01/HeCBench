# blockexchange: verified four-model study observation

RQ IDs: RQ1–RQ5. Experiment IDs: E1–E5. Evidence ID: 20261017T000905953305Z_blockexchange.
Configuration contract: `{"n": 4096, "repeat": 10, "threads": 256, "launch_blocks": 16, "items_per_thread": 4, "tile_elements": 1024, "fixture": "Int32i0..n-1", "outputs": "all4096permutationresults exact Float64encoding", "timing": "synchronizedwholekernelmean, originaloverprovisionedgridretained"}`.
Protocol: five_trials_last_four_v1, all five correctness checks retained, timing mean trials2–5.
Evidence: `/home/imo/HeCBench_julia/.porting-state/portable_julia/runs/20261017T000905953305Z_blockexchange/result.json`, config.json, timings.csv, raw logs, binary artifacts.
Ontology: ontology/benchmarks/blockexchange.yaml; SHA256 `3657b3c5b2debdc13a3061da277a7c9870bae10a359ccd1ce39de72a6c1dbc8b`. Original inferred ontology labels retained; no retroactive winner-based relabeling.

## Numerical confirmation
Each variant passes independent full-element CPU oracle checks and direct binary-array comparisons against native CUDA and CUDA.jl on every one of the five trials. Acceptance tolerances and all observed absolute/relative errors are preserved in result.json. NaN/nonfinite and shape/count checks are performed by the runner. Native device kernels are included unchanged; historical CUDA.jl original_main.jl and original_Makefile are retained. Parent compatibility entrypoints execute the historical original source. Study drivers change only explicitly recorded reference inputs/measurement handling, a separate phase from original translation.

The actual blocked-to-striped exchange is retained: four items per thread, 256 threads, 1024-element tiles, blocked loads into shared memory, a workgroup barrier, striped shared reads and blocked stores. All4096 output elements match a full independent CPU permutation and both GPU references exactly across five trials. The original gridceil(n/256)=16 is retained, although only four blocks have active tiles. Historical CUDA.jl direct one-item-per-thread gathering is separately corrected to the native shared four-item mapping. Native BlockLoad/BlockStore helpers versus explicit scalar transfers and JACC extra1024-zero shared copy remain implementation confounds. JACC copies its workspace only in active blocks and reserves4096 shared bytes. Partial tiles are untested. Initial native build failed because the study omitted the original Makefile include paths for blockAccess and warpexchange headers; the second build restores those existing paths and records complete dependencies, without modifying native kernels. KA/JACC each succeed on their first source candidate; native adapter build repair is a separate phase.

## Runtime: microseconds
Each trial records the synchronized launch-loop mean per internal repeat. This includes host dispatch and excludes JIT/compilation, setup, allocations, transfers, input reset, CPU checking and artifact writing. Each trial starts with the same inputs. A device warmup precedes trial1. All timing commands are serialized on GPU6 via the shared lock. Native CUDA, CUDA.jl, KA and JACC are measured under the same named contract. Julia environments pin CUDA runtime12.0; exact package/hardware evidence lives in setup and run bundles.

| Model | Trials1–5 (us) | Mean2–5 (us) |
|---|---|---|
| cuda | 3.6400, 3.7240, 3.6180, 3.5690, 3.6020 | 3.6282 |
| cuda_jl | 5.3231, 7.3371, 4.6461, 4.4430, 4.3040 | 5.1825 |
| ka | 10.4121, 17.2761, 8.0071, 7.5680, 7.1860 | 10.0093 |
| jacc | 14.2211, 15.6372, 10.3861, 9.7991, 9.6721 | 11.3736 |

## Attempts, tokens and feature support
Each new framework has one source candidate; zero compiler-feedback code repairs. Actual runner build/JIT invocations are recorded separately from timing trials; tool/reuse/setup failures are infrastructure events, not translation candidates. A five-trial timing series is not five translation attempts. Token accounting is partial source-text characters/4, with unavailable conversation/tool/reasoning/cached usage explicitly unknown; these estimates do not establish total billed cost or an abstraction cost ranking. Shared infrastructure/reference correction/paper work are separate phases. KA-first and JACC reuse make this incremental production effort, not independent translations. No unsupported framework feature was observed for this selected configuration.

## Ontology explanation and paper follow-up
Use the existing operation/access classifications as grouping evidence, with implementation choices above recorded separately. Any runtime gap is an observation. Potential launch/configuration overhead, memory access behavior, compiler output and changed specialization are competing explanations, not demonstrated causes. No measured memory traffic, occupancy or roofline position was collected in this pass. Profiling and size scaling remain pending; this small configuration cannot support broad abstraction-selection recommendations. RQ1 is answered for this contract; RQ2 cost completeness is partial; RQ3 has matched measurements; RQ4 mechanisms and RQ5 generalization remain unresolved. Coordinator must export a portable snapshot and connect this observation to claims/figures/sections.
