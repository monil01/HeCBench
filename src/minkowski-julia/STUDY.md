# minkowski: scientific and implementation observation

The sourcecomputesMinkowski norms despite itsmatrixproductcomment. Studyretainsallfourp cases andorderedfeatureaccumulation withnative16x16tile. Named64x128x128template usesnonuniformdyadicinputs toexerciserowandcolumnindexing; speedratios arenotdefaultsizeperformanceclaims.

## Reference audit

Historical Julia runs256-element touch_kernel add1 and prints hardcoded distance PASS; no pairwise norm calculation. Faithful study reference implements native row/column featureloop andnormpowers.

## Numerical and performance evidence

Configuration `minkowski_m64_n128_k128_allp1to4_r10_v1`, result `/home/imo/HeCBench_julia/.porting-state/portable_julia/runs/20261017T162209712310Z_minkowski/result.json`. Five in-process trials per model; full outputs checked in every trial against both references; mean trials2–5. Warmup precedes trial1. Native kernel unchanged. This named configuration does not certify all CLI modes or physical model validity.

## Cost and failures

One saved candidate and one compiler/JIT process per target model; no observed target repair. Historical CUDA.jl attempts unknown. Reference S1 semantic repair separate from target translation. Exact consumed tokens unavailable; final-source proxies are partial, never total costs.

Reproduce: `python3 tools/verify_portable_julia.py --config src/minkowski-julia/study_config_ka.json`, then JACC config reusing saved references. Do not run a new series merely to refresh metadata.
