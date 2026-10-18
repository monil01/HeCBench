# ludb: scientific and implementation observation

The study compares framework allocation and vendor-library interoperability for dimension-48 partial-pivot batched LU. Four matrices cover diagonal, nontrivial row permutation, singular zero, and signed dense diagonal-dominant inputs. Complete factors, pivots and info/status outputs are compared against both references in every capture. Independent exact-rational multiplication of the actual Float32 factors checks PA=LU including zero reference entries. This verifies the named factorization contract but does not measure a framework-authored LU kernel or establish solver behavior for arbitrary conditioning.

## Reference audit

Historical CUDA.jl genuinely uses cuBLAS LU, but calls strided-batched wrapper with allocation of pivot/info outputs within ROI, whereas native uses pointer-array batched API with preallocated outputs. The study aligns raw pointer-array API, preallocation, pinned library and context-wide completion for all models. RNG inputs differ historically; all four consume an identical fully initialized named fixture. No historical wrong LU computation is established.

## Numerical and performance evidence

Configuration `ludb_original48_batch4_pivot_singular_fullfactors_v1`, result `/home/imo/HeCBench_julia/.porting-state/portable_julia/runs/20261018T003810701750Z_ludb/result.json`. Five in-process trials per model; full outputs checked in every trial against both references; mean trials2–5. Warmup precedes trial1. Native kernel unchanged. This named configuration does not certify all CLI modes or physical model validity.

## Cost and failures

One interoperability program candidate per model, with zero authored device kernel definitions and zero authored-kernel compiler checks. CUDA.jl reference preparation had two host program validations after one shared harness B2 whitespace parse repair; portable versions each one program validation if successful. Prospective counters describe full activity windows, not exclusive coding cost. Historical CUDA.jl attempts unknown. Reference correction or documented P1 mapping phase is separate from target translation. Observed prospective usage intervals are recorded separately; exclusive billed translation cost is unknown. Final-source proxies remain partial.

Reproduce: `python3 tools/verify_portable_julia.py --config src/ludb-julia/study_config_ka.json`, then JACC config reusing saved references. Do not run a new series merely to refresh metadata.

## Library and numerical scope

All three Julia models are vendor-library interoperability with zero authored kernels; each framework actually allocates its device storage. The original native getrfBatched wrapper is unchanged. A pinned system cuBLAS is used throughout, and context-wide synchronization prevents raw-handle stream completion errors. All 9,216 factor entries, 192 pivot entries, four info values and the API status are compared exactly, with 9,221 actual input/control values checked byte-for-byte. Float64 artifact encoding and rational host checking do not imply FP64 GPU factorization.

The independent witness additionally applies the original 1e-6 relative bound at every nonzero reference entry and checks zero-entry residuals. The singular zero case must return info=1. Named small-batch correctness does not certify arbitrary conditioning or default batch10,000 performance. KA usage includes same-benchmark metadata drafting while verification queued; all counters are activity windows rather than exclusive coding costs.


Prospective rawusage intervals include cached/repeatedcontext and coordination; the parentworkflow contains its childphases and must not be summed with them. Exclusive billed translation cost remains unknown; source-size proxies remain separately labeled.
