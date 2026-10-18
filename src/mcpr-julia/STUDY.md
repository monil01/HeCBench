# mcpr: scientific and implementation observation

This benchmark estimates categorical choice frequencies: each sample adds Gaussian noise to a row of 21 utility means, selects a maximum, and increments that category. We compare all three source layouts, including global private workspace versus per-thread columns in shared memory. Every output probability is checked against an independent integer winner count and the probability-simplex identity. The finite shared sample is fixed across all models and reused across parameter rows. Agreement certifies that empirical operator for these inputs, not the error of the probability estimate relative to a Gaussian population or Monte Carlo convergence. Separate deterministic tie controls verify the actual policy: the final category wins a tied maximum; otherwise the earliest maximizing category wins.

## Reference audit

HistoricalJulia launchesoneInt32touchthread andprintsPASS withgenericstructuralmessages, performingnoGaussianutilities,categoryargmax,MonteCarlo countsorprobabilities. Entireoriginalsharedsmokestubpreserved innestedoriginal_main.jl.

## Numerical and performance evidence

Configuration `mcpr_three_layouts_n193_m97_gaussian_n5_m3_ties_fullprobabilities_v1`, result `/home/imo/HeCBench_julia/.porting-state/portable_julia/runs/20261018T000229132785Z_mcpr/result.json`. Five in-process trials per model; full outputs checked in every trial against both references; mean trials2–5. Warmup precedes trial1. Native kernel unchanged. This named configuration does not certify all CLI modes or physical model validity.

## Cost and failures

One authored device candidate for each Julia model. The first CUDA.jl program validation failed while parsing the shared host oracle because a generator lacked whitespace before for; no device JIT had occurred. The second validation fixed shared instrumentation only and compiled all three unchanged authored kernels. KA and JACC each validate one program candidate with all three kernels. Native source kernels and source1e-3 tolerance remain unchanged; exact recovered counts are an additional stronger check. Historical CUDA.jl attempts unknown. Reference correction or documented P1 mapping phase is separate from target translation. Observed prospective usage intervals are recorded separately; exclusive billed translation cost is unknown. Final-source proxies remain partial.

Reproduce: `python3 tools/verify_portable_julia.py --config src/mcpr-julia/study_config_ka.json`, then JACC config reusing saved references. Do not run a new series merely to refresh metadata.

## Exact empirical operator and source-input scope

Every capture contains all three complete probability matrices. An independent eligible-set argmax oracle recovers every integer count exactly and verifies nonnegative probabilities summing to one. Separate zero-noise controls give winners [20,0,20,4,20], testing both last-category tie priority and earliest-category priority when the last category is below the maximum. The Gaussian fixture includes a realized dominant row; its sample-wise dominance is checked directly, without claiming a population probability of exactly one. Noise vectors are shared across parameter rows. Finite empirical agreement does not establish Gaussian probability-estimation error or Monte Carlo convergence.

The original CLI reads a comma-separated file with repeated fscanf("%lf") calls. A minimal CPU witness reads the first value, then returns zero at the comma without advancing or writing the second destination. The named study supplies fully initialized generated arrays directly to unchanged native kernels, bypassing that reader. This documents an unmeasured CLI/input-loader limitation; it is not a failure demonstrated by the measured kernel inputs. The original data file is neither changed nor required by the study. The frozen shared fixture and generator are committed source assets.

Private AoS and SoA kernels keep the native 192-thread geometry; the shared variant keeps 96 threads and 32,256 bytes. Julia specializes the source's fixed K=21 and unrolls private utility construction, whereas native K is a runtime kernel parameter. That specialization and private-array/tuple representation are explicit mapping differences. JACC initializes and copies its full shared template and executes an initial block barrier before guarding partial rows; the source algorithm itself has no inter-thread communication or required barrier. No unprofiled register, spill or bottleneck attribution is made.

## Component performance

Five in-process trials, discard the first and average the last four for each component. Units below are microseconds for the 193-row Gaussian case; the smaller tie-control timings are retained in structured evidence, together with all five raw component series. Each component is a mean of three kernel launches. The headline benchmark time is the sum of all six case/layout component means.

| Model | AoS private | SoA private | SoA shared |
|---|---:|---:|---:|
| cuda | 304.261 | 259.073 | 204.885 |
| cuda_jl | 186.640 | 104.193 | 219.146 |
| ka | 190.718 | 109.699 | 223.464 |
| jacc | 188.523 | 109.532 | 232.762 |


Prospective rawusage intervals include cached/repeatedcontext and coordination; the parentworkflow contains its childphases and must not be summed with them. Exclusive billed translation cost remains unknown; source-size proxies remain separately labeled.

Prospective KA activity also includes the coordinator-requested handoff of finalized SPD2S/Keccak commit commands and a read-only inspection of commit-helper arguments. No additional translation or GPU work for those benchmarks occurred. Counters are retained without correction and do not measure exclusive coding cost.
