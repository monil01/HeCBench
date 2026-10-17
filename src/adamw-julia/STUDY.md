# adamw: scientific and implementation observation

An exhaustive independent nibble roundtrip witness establishes a common native CPU/GPU defect: right quantization indices0..15 are ANDed with0xf0 without shifting, so15of16 cannot roundtrip. These ports preserve that actual behavior for native comparison and cannot be interpreted as correct general quantized AdamW. The fixture starts with nonzero upper nibbles and checks all mutable parameters, bytes and scales across two steps. The historical unquantized Julia implementation is separately corrected. Full64threadblocks avoid the native partial-block collective hazard.

## Reference audit

Historical CUDA.jl computes ordinary unquantized scalar AdamW, not native packed four-bit moments, per-block scales or quantization. Repaired reference reproduces actual native algorithm including established upper-nibble packing defect.

## Numerical and performance evidence

Configuration `adamw_native_pack_quirk_n4096_steps2_sharedtree_v1`, result `/home/imo/HeCBench_julia/.porting-state/portable_julia/runs/20261017T172922812638Z_adamw/result.json`. Five in-process trials per model; full outputs checked in every trial against both references; mean trials2–5. Warmup precedes trial1. Native kernel unchanged. This named configuration does not certify all CLI modes or physical model validity.

## Cost and failures

One saved candidate and one compiler/JIT process per target model; no observed target repair. Historical CUDA.jl attempts unknown. Reference correction or documented P1 mapping phase is separate from target translation. Exact consumed tokens unavailable; final-source proxies are partial, never total costs.

Reproduce: `python3 tools/verify_portable_julia.py --config src/adamw-julia/study_config_ka.json`, then JACC config reusing saved references. Do not run a new series merely to refresh metadata.
