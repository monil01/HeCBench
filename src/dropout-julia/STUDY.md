# dropout: scientific and implementation observation

A documented historical hash-RNG portability compromise prevents reproducibility even when dropout algebra is retained. Exact counter layout, cached offset lane and variant-specific random draw assignment are part of this scientific contract. All 6,303,840 output mask bits and scaled Float32 bits agree exactly across four versions, including negative zero. Source VEC2 intentionally discards two of each four random values. Vector PTX inspection is separate from measured timing and does not establish bandwidth or performance causality.

## Reference audit

Historical CUDA.jl implements dropout using a documented deterministic hash RNG portability substitution rather than native seeded cuRAND Philox. This changes exact seeded masks/draw mapping, not an established violation of the Bernoulli operator law. Current reference restores exact Philox counter/subsequence/offset STATE1 and source VEC-specific draw assignment using own Julia integer arithmetic. Native unchanged.

## Numerical and performance evidence

Configuration `dropout_original_grids_1028_524292_Philox_allp_timedp4_replayp123_v1`, result `/home/imo/HeCBench_julia/.porting-state/portable_julia/runs/20261018T051652412401Z_dropout/result.json`. Five in-process trials per model; full outputs checked in every trial against both references; mean trials2–5. Warmup precedes trial1. Native kernel unchanged. This named configuration does not certify all CLI modes or physical model validity.

## Cost and failures

CUDA.jl three whole-program checks: shared parse failure before JIT, authored tuple-capture InvalidIR failure, then pass; two authored device-source candidates/JIT processes. One shared Python fixture namespace failure separate. KA/JACC first candidate/process pass. Mapping inspection adds one separate compilation process and two untimed launches per Julia model, no timing trials. Historical CUDA.jl attempts unknown. Reference correction or documented P1 mapping phase is separate from target translation. Prospective harness activity counters retained, including cached/repeated context and coordination; historical usage unknown and source proxies partial, not billed or exclusive coding cost.

Reproduce: `python3 tools/verify_portable_julia.py --config src/dropout-julia/study_config_ka.json`, then JACC config reusing saved references. Do not run a new series merely to refresh metadata.

## Capture and mapping scope

All20 artifacts bitidentical, each13,134,263 UInt32 words:525345 exact prefix +86 rawprobe/guards +6303840 scaledoutputbitwords +6303840 masks +1152 guards. Every trial checks every mask and Float32 scaled bit against independent Python integer Philox oracle, including negative zero. Actual timed p4 state is copied before p1..3 untimed deterministic replays; those replay arrays are not claimed original timed intermediate states. Raw80 RNG probe once per process, repeated in capture records, not five independent executions. Source VEC2 discards random z/w lanes exactly as original. Emitted PTX confirms packed .v2/.v4 payload loads/stores for all4, separate untimed inspection with source hashes, no traffic/roofline causal claim. Guarded payload16B alignment differs from original cudaMalloc256B payload alignment. Means seconds {"cuda": 0.00028960250000000003, "cuda_jl": 0.00043579925000000004, "ka": 0.0005670795, "jacc": 0.0008235755}. All native outliers retained. Two authored benchmark definitions/three requested instances plus one verification-only RNG probe do not imply compiler invocation counts.
