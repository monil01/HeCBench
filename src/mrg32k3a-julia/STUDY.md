# mrg32k3a: verified four-model study observation

RQ IDs: RQ1–RQ5. Experiment IDs: E1–E5. Evidence ID: 20261016T233455267658Z_mrg32k3a.
Configuration contract: `{"precision": "Float32", "size": 4096, "seed": 1234, "offset": 0, "rng": "CURAND_RNG_PSEUDO_MRG32K3A", "ordering": "CURAND_ORDERING_PSEUDO_BEST", "internal_repeats": 1, "timing": "deviceendtoend allocation/create/configure/generate/D2H/release/destroy; CPUgeneratorseparatetimingauxiliary", "outputs": "4096deviceuniformnumbers comparedhostcuRANDreferenceandbothGPUrefs"}`.
Protocol: five_trials_last_four_v1, all five correctness checks retained, timing mean trials2–5.
Evidence: `/home/imo/HeCBench_julia/.porting-state/portable_julia/runs/20261016T233455267658Z_mrg32k3a/result.json`, config.json, timings.csv, raw logs, binary artifacts.
Ontology: ontology/benchmarks/mrg32k3a.yaml; SHA256 `7802ebfcbdca848e93547f0e072d797966217a9ff2ebf35a2d20a6b5fd6dbcf0`. Original inferred ontology labels retained; no retroactive winner-based relabeling.

## Numerical confirmation
Each variant passes full-element host-cuRAND checks (same vendor algorithm, not an independent implementation) and direct binary-array comparisons against native CUDA and CUDA.jl on every one of the five trials. Acceptance tolerances and all observed absolute/relative errors are preserved in result.json. NaN/nonfinite and shape/count checks are performed by the runner. Native device kernels are included unchanged; historical CUDA.jl original_main.jl and original_Makefile are retained. Parent compatibility entrypoints execute the historical original source. Study drivers record reference input alignment, any algorithm correction, and measurement handling as separate phases from the historical translation.

Vendor-library interoperation stratum: ZERO translated device kernels. KA allocates via KernelAbstractions.allocate CUDAbackend, JACC via JACC.zeros; both explicitly call cuRAND MRG32K3A through CUDA.CURAND. n4096 seed1234 offset0 orderingPSEUDO_BEST, all4096 outputs exact acrossmodels, each also compared hostcuRAND within1e-3. HostcuRAND is independentdeviceexecution, NOT independent RNGalgorithm oracle. Do not count as pure framework kernel translation. Timedregion explicitly includes deviceallocation, generatorcreation/configuration, generation, D2H, release and generator destruction; auxiliaryCPUgenerator times separately. Native generator cleanup regularized against original leaked handles, nativecudaMalloc/free versusJuliamemorypoolrelease, JACCzerosinitialization versusKAundef allocation confounds; native/Julia vendorlibrary paths/versions pending explicitcheck.

## Runtime: microseconds
Five in-process trials are retained; the reported mean excludes trial1 and averages trials2–5. Each value uses the configuration's exact timed region and normalization. Notes above and `input_contract` distinguish per-repeat means from case or pipeline totals. Transfers, resets, allocation, temporary storage, graph control, host dispatch and synchronization are included where that named contract specifies. Compiler/JIT warmup, CPU correctness and artifact writing occur outside measurement. All GPU measurements use the shared GPU6 lock. Exact package/hardware evidence lives in setup and run bundles.

| Model | Trials1–5 (us) | Mean2–5 (us) |
|---|---|---|
| cuda | 3150.2400, 4695.0140, 7492.7610, 3417.9630, 3285.0910 | 4722.7072 |
| cuda_jl | 2893.0180, 3711.7650, 3051.3490, 3128.3990, 3043.0690 | 3233.6455 |
| ka | 2509.0640, 3139.6600, 2400.9520, 3208.4200, 2438.4830 | 2796.8787 |
| jacc | 2609.8840, 2609.0540, 2651.9040, 2945.9470, 2589.5540 | 2699.1147 |

## Attempts, tokens and feature support
Each new framework has one source candidate; zero compiler-feedback code repairs. Actual runner build/JIT invocations are recorded separately from timing trials; tool/reuse/setup failures are infrastructure events, not translation candidates. A five-trial timing series is not five translation attempts. Token accounting is partial source-text characters/4, with unavailable conversation/tool/reasoning/cached usage explicitly unknown; these estimates do not establish total billed cost or an abstraction cost ranking. Shared infrastructure/reference correction/paper work are separate phases. KA-first and JACC reuse make this incremental production effort, not independent translations. No unsupported framework feature was observed for this selected configuration.

## Ontology explanation and paper follow-up
Use the existing operation/access classifications as grouping evidence, with implementation choices above recorded separately. Any runtime gap is an observation. Potential launch/configuration overhead, memory access behavior, compiler output and changed specialization are competing explanations, not demonstrated causes. No measured memory traffic, occupancy or roofline position was collected in this pass. Profiling and size scaling remain pending; this small configuration cannot support broad abstraction-selection recommendations. RQ1 is answered for this contract; RQ2 cost completeness is partial; RQ3 has matched measurements; RQ4 mechanisms and RQ5 generalization remain unresolved. Coordinator must export a portable snapshot and connect this observation to claims/figures/sections.

## Vendor library provenance
Both native and Julia cuRAND resolve to `/usr/lib/x86_64-linux-gnu/libcurand.so.10.3.1.124`, APIversion10301. Postmeasurement libraryquery logged separately; no additional timingseries. This removes vendorversiondifference as a confound for this measured setup.
