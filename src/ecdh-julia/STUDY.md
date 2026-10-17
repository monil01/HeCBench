# ecdh: verified four-model study observation

RQ IDs: RQ1–RQ5. Experiment IDs: E1–E5. Evidence ID: 20261017T002300270879Z_ecdh.
Configuration contract: `{"n": 4096, "repeat": 10, "secret": 18, "base_point": [5, 1], "modulus": 17, "curve_A": 2, "outputs": "slowx,slowy,fastx,fasty arrays all4*n exactFloat64encoding", "oracle": "Independentfinitefieldgroupaddition withidentityandinverse (Julia invmod/nativebruteforceinverse), computes18P", "timing": "sumslow/fast synchronizedloopmeans"}`.
Protocol: five_trials_last_four_v1, all five correctness checks retained, timing mean trials2–5.
Evidence: `/home/imo/HeCBench_julia/.porting-state/portable_julia/runs/20261017T002300270879Z_ecdh/result.json`, config.json, timings.csv, raw logs, binary artifacts.
Ontology: ontology/benchmarks/ecdh.yaml; SHA256 `8903e24189eadbe220053addc76be6fbebc87572ec885edca297ebca7f309dd8`. Original inferred ontology labels retained; no retroactive winner-based relabeling.

## Numerical confirmation
Each variant passes independent full-element CPU oracle checks and direct binary-array comparisons against native CUDA and CUDA.jl on every one of the five trials. Acceptance tolerances and all observed absolute/relative errors are preserved in result.json. NaN/nonfinite and shape/count checks are performed by the runner. Native device kernels are included unchanged; historical CUDA.jl original_main.jl and original_Makefile are retained. Parent compatibility entrypoints execute the historical original source. Study drivers change only explicitly recorded reference inputs/measurement handling, a separate phase from original translation.

Both native algorithms are retained: slow repeated point addition and fast double-and-add, with original secret18, basepoint(5,1), modulus17 and A2. Every thread performs the original arithmetic rather than loading an expected result. All16,384 coordinate outputs match CPU and both references exactly across five trials atn4096,r10. Correctness is strengthened beyond native slow-versus-fast equality by an independent finite-field group oracle with an explicit identity element and inverse operation: Julia uses invmod, the native adapter brute-forces the tiny inverse field. The original all-thread duplicate secret18 input is retained; varying keys and other branches are untested. Both frameworks succeed on candidate one with two kernel specializations and no shared-memory requirement.

## Runtime: microseconds
Each trial records the synchronized launch-loop mean per internal repeat. This includes host dispatch and excludes JIT/compilation, setup, allocations, transfers, input reset, CPU checking and artifact writing. Each trial starts with the same inputs. A device warmup precedes trial1. All timing commands are serialized on GPU6 via the shared lock. Native CUDA, CUDA.jl, KA and JACC are measured under the same named contract. Julia environments pin CUDA runtime12.0; exact package/hardware evidence lives in setup and run bundles.

| Model | Trials1–5 (us) | Mean2–5 (us) |
|---|---|---|
| cuda | 34.0813, 34.2993, 34.3013, 34.1943, 34.0923 | 34.2218 |
| cuda_jl | 45.6144, 6094.7539, 46.1074, 45.5983, 45.4623 | 1557.9805 |
| ka | 49.5115, 54.5744, 46.0883, 46.3053, 45.6333 | 48.1503 |
| jacc | 52.8255, 57.1006, 48.1244, 47.1905, 47.5245 | 49.9850 |

## Attempts, tokens and feature support
Each new framework has one source candidate; zero compiler-feedback code repairs. Actual runner build/JIT invocations are recorded separately from timing trials; tool/reuse/setup failures are infrastructure events, not translation candidates. A five-trial timing series is not five translation attempts. Token accounting is partial source-text characters/4, with unavailable conversation/tool/reasoning/cached usage explicitly unknown; these estimates do not establish total billed cost or an abstraction cost ranking. Shared infrastructure/reference correction/paper work are separate phases. KA-first and JACC reuse make this incremental production effort, not independent translations. No unsupported framework feature was observed for this selected configuration.

## Ontology explanation and paper follow-up
Use the existing operation/access classifications as grouping evidence, with implementation choices above recorded separately. Any runtime gap is an observation. Potential launch/configuration overhead, memory access behavior, compiler output and changed specialization are competing explanations, not demonstrated causes. No measured memory traffic, occupancy or roofline position was collected in this pass. Profiling and size scaling remain pending; this small configuration cannot support broad abstraction-selection recommendations. RQ1 is answered for this contract; RQ2 cost completeness is partial; RQ3 has matched measurements; RQ4 mechanisms and RQ5 generalization remain unresolved. Coordinator must export a portable snapshot and connect this observation to claims/figures/sections.
