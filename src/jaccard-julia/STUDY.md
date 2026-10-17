# jaccard: scientific and implementation observation

Graph Jaccard requires graphstructure and weightedintersection correctness, which the historical fill-only Julia source did not compute. New versions preserve the actual five-kernel pipeline in both weighted/unweighted modes. Warp16 scan and atomic operations are explicit CUDA backend extensions, so the KA/JACC versions form a hybrid device-intrinsic stratum. Independent CPU set intersection checks complete outputs/intermediates. The generated64vertex graph doesnotestablish application-level graphquality or general empty-row behavior.

## Reference audit

HistoricalJuliaonlyfillswork[i]=i andprints timings, nevercomputesCSRJaccardsimilarity. Repairedreferenceimplementstrue5kernelweighted/unweightedpipeline, originalCSR/order/geometry/warp16scan/optimizedmerge/atomic preserved.

## Numerical and performance evidence

Configuration `jaccard_n64_cycliccsr_degrees17to31_weighted_unweighted_v1`, result `/home/imo/HeCBench_julia/.porting-state/portable_julia/runs/20261017T182833630610Z_jaccard/result.json`. Five in-process trials per model; full outputs checked in every trial against both references; mean trials2–5. Warmup precedes trial1. Native kernel unchanged. This named configuration does not certify all CLI modes or physical model validity.

## Cost and failures

CUDA.jl firstprocess stoppedbeforedeviceJIT due sharedCPUoracle syntax; one actualdevicecandidate, two processchecks, one sharedhostadapterrepair. KA/JACCcounts recorded independently. Historical CUDA.jl attempts unknown. Reference correction or documented P1 mapping phase is separate from target translation. Observed prospective usage intervals are recorded separately; exclusive billed translation cost is unknown. Final-source proxies remain partial.

Reproduce: `python3 tools/verify_portable_julia.py --config src/jaccard-julia/study_config_ka.json`, then JACC config reusing saved references. Do not run a new series merely to refresh metadata.


A separate dependency-free Python set-intersection witness checked all9212saved values in each of20model/trial artifacts, including finalweighted/unweightededgeweights andallrow/intersection/sumvolumes. It uses set operations rather than the device two-pointer merge. Maximum absoluteoracleerror is 1.1920928955078125e-06. Every cross-model output is bitwiseidentical in this configuration.


Prospective rawusage intervals include cached/repeatedcontext and coordination; the parentworkflow contains its childphases and must not be summed with them. Exclusive billed translation cost remains unknown; source-size proxies remain separately labeled.
