# randomAccess: scientific and implementation observation

An independent serial traversal of the original LFSR stream predicts every final UInt64tableword, while GPUversions retain128jumpaheadsubstreams and atomicXOR. Exactcomparison strengthens the original allowed1percent residual criterion. Smaller tableextent changes cachelocality andcollisionrates, so performance describes this named random-access regime. No floatingpoint or scientificfieldaccuracy is involved.

## Reference audit

Historical Julia only touches a256-element Float32array and prints fixed0errors/PASS, without RNG or atomic table updates. Genuine UInt64HPCCjumpahead plus atomicXOR nowimplemented.

## Numerical and performance evidence

Configuration `randomaccess_table2p14_updates2p16_128threads_r10_v1`, result `/home/imo/HeCBench_julia/.porting-state/portable_julia/runs/20261017T173954656341Z_randomAccess/result.json`. Five in-process trials per model; full outputs checked in every trial against both references; mean trials2–5. Warmup precedes trial1. Native kernel unchanged. This named configuration does not certify all CLI modes or physical model validity.

## Cost and failures

One saved candidate and one compiler/JIT process per target model; no observed target repair. Historical CUDA.jl attempts unknown. Reference correction or documented P1 mapping phase is separate from target translation. Exact consumed tokens unavailable; final-source proxies are partial, never total costs.

Reproduce: `python3 tools/verify_portable_julia.py --config src/randomAccess-julia/study_config_ka.json`, then JACC config reusing saved references. Do not run a new series merely to refresh metadata.
