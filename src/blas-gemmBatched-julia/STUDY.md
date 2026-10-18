# blas-gemmBatched: scientific and implementation observation

A benchmark named GEMM actually measures square matrix-vector products (n=1), and its historical Julia entry redirects to a different scientific algorithm. Genuine restoration uses all three original precision APIs. Complete typed outputs, exact inactive rows/padding/guards and independent bounded dots agree in all four models. Identical pinned vendor computation and zero authored kernels make this a separate interoperability stratum; do not infer a framework kernel performance advantage.

## Reference audit

Historical CUDA.jl is only include("../mcpr-julia/main.jl"), an unrelated Monte Carlo photon-transport program, not native batched matrix-vector GEMM. Original bytes retained. Restore actual H/S/DgemmBatched vendor APIs, matrix/vector pointer lists, padded leading dimensions and complete size sweeps with strict full-output checks. Native operator/API unchanged; guarded named harness uses matching preallocation/context completion.

## Numerical and performance evidence

Configuration `gemmBatched_HSD_all_sizes2_17_batches3_17_twofixtures_pinned120_v1`, result `/home/imo/HeCBench_julia/.porting-state/portable_julia/runs/20261018T035020924763Z_blas-gemmBatched/result.json`. Five in-process trials per model; full outputs checked in every trial against both references; mean trials2–5. Warmup precedes trial1. Native kernel unchanged. This named configuration does not certify all CLI modes or physical model validity.

## Cost and failures

One whole-program candidate/check per Julia model; all pass. One separate native build-argument B2 precedes native source compilation and is fixed by standard -L/-l spelling, not by changing the native algorithm. Shared reference/oracle/library ABI work precedes KA then JACC. Historical CUDA.jl attempts unknown. Reference correction or documented P1 mapping phase is separate from target translation. Actual prospective harness activity windows retained per phase with repeated/cached context and lag; nonexclusive, historical consumed usage unknown. Saved-source proxies partial.

Reproduce: `python3 tools/verify_portable_julia.py --config src/blas-gemmBatched-julia/study_config_ka.json`, then JACC config reusing saved references. Do not run a new series merely to refresh metadata.

## Complete typed vendor contract

Each capture:36792exactFloat64-encoded actual typed input values/metadata +32640completeoutputelements (18240active,14400padding) +384outputguards +192statusvalues=70008values. All20completeartifactbytes match native trial1 exactly. Independent Decimal90 exact-input dots also satisfy every original predeclared bound, with no acceptance or timing edits. Native/Juliapaths additionally compare all typed A/B input bytes and complete pointer lists including guards exactly; pointer addresses are process-specific and not cross-compared. All three programs use actual model-specific allocation with raw pinned cuBLAS calls; authoreddevicekernels=0, library-interoperability stratum.

Mean benchmark pipeline seconds: {"cuda": 0.00230996005, "cuda_jl": 0.0028910794249999974, "ka": 0.0031102366250000024, "jacc": 0.003050595450000001}. Originalupper100/batch25000CLI not measured.
