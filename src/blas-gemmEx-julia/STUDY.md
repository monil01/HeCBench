# blas-gemmEx: scientific and implementation observation

These versions target a vendor GEMMEx benchmark, not handwritten devicekernels. Same localcuBLASlibrary, data/compute enums androwmajoroperands allow directoutputcomparison. NativeFP32mode labels do not select their claimed modes because switchfallthrough restoresFP32; actualmapping isrecorded. All nine named requests return success, including BF16; the internal hardware path is not established. Exactdyadicinputs support an independent integer-dot-derived oracle, but do notstress generalmixedprecision rounding.

## Reference audit

HistoricalJuliahasgenuineGEMMbut column-majorreshapeinputlayout andscalarint8path differfrom native row-major cuBLASGemmEx contract. NativeFP32switchlacksbreaks, makingallfour labeledcomputemodes FP32; preservedanddocumented asprecisionintent deviation. Newreferenceuses exactsame localcuBLASlibrary/API androwmajoroperands.

## Numerical and performance evidence

Configuration `gemmex_m32_n32_k16_dyadic_all9requests_localcublas120_v1`, result `/home/imo/HeCBench_julia/.porting-state/portable_julia/runs/20261017T180031142251Z_blas-gemmEx/result.json`. Five in-process trials per model; full outputs checked in every trial against both references; mean trials2–5. Warmup precedes trial1. Native kernel unchanged. This named configuration does not certify all CLI modes or physical model validity.

## Cost and failures

CUDA.jl, KA and JACC each require one saved variant candidate and one process verification. Native shared preparation had one adapter compile repair, followed by one scientifically motivated output/residency instrumentation correction; both phases retained separately. Prospective usage counters exist for shared preparation, KA, shared measurement correction, and JACC; prior source audit unmetered, repeated/cache context included. Historical CUDA.jl attempts unknown. Reference correction or documented P1 mapping phase is separate from target translation. Exact consumed tokens unavailable; final-source proxies are partial, never total costs.

Reproduce: `python3 tools/verify_portable_julia.py --config src/blas-gemmEx-julia/study_config_ka.json`, then JACC config reusing saved references. Do not run a new series merely to refresh metadata.


All nine requested cases actually return cuBLAS success, including BF16 on the V100; no unsupported case is excluded. API success does not establish which internal GPU algorithm or hardware arithmetic path was used. Native and Julia runs report the identical library path/version12.0.2.224/120002 and exact9,225-value artifacts in every trial. Native managed buffers are prefetched to the device before each primary trial and output checks use explicit D2H copies; the earlier uncontrolled-residency series remains archived as nonprimary instrumentation evidence. Prospective agent-counter intervals are recorded separately for KA, JACC, shared reference preparation, and shared measurement correction. Those observations include repeated/cache context and are not exclusive billed costs; source audit predates the first boundary, and shared preparation includes one disclosed unrelated documentation tool round.

## Completion/ordering measurement correction

RawcuBLAS defaultlegacy0 handle completion is now device-wide with CUDA.device_synchronize(blocking=true). Native unchanged five captures reused; allthreeJulia five fresh. Oldnumericalcaptures retained, oldtimingexcluded before replacement. No observed numerical failure or quantified timing bias inferred from missing ordering/completion proof. Not a translation or tuning attempt.

Corrected configuration `gemmex_m32_n32_k16_dyadic_all9requests_localcublas120_v1__devicewide_completion_v2`, result `.porting-state/portable_julia/runs/20261017T224917062565Z_blas-gemmEx/result.json`. All five correctness checks against both references pass. Time1 alone is discarded; mean times2–5 in seconds: cuda=0.000168047475, cuda_jl=0.00014147742500000002, ka=0.0001486065, jacc=0.0001470015. Original translation attempts/tokens and commit receipts remain distinct from this shared measurement repair.
