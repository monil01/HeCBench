# silu: scientific and implementation observation

Includes allfive native SiLU variants at256threads, preserves4lanevectorworkitems and bothderivativeformulas. Named signedfixture exercises positive/negative activations andgradients. Any timing difference is scoped to thissmalltensor andhostlaunch loop; noapplication speedupor vectorinstructionidentity claimed.

## Reference audit

Historical CUDA.jl forward2 and backward3 labels launch scalar kernels instead of genuine native four-element vector work items. Study repairs these cases with NTuple4 reinterpret arrays and 4lanes perworkitem. Scalar formulas retained.

## Numerical and performance evidence

Configuration `silu_all5_fp32_n4096_bs256_r10_v1`, result `/home/imo/HeCBench_julia/.porting-state/portable_julia/runs/20261017T161403595465Z_silu/result.json`. Five in-process trials per model; full outputs checked in every trial against both references; mean trials2–5. Warmup precedes trial1. Native kernel unchanged. This named configuration does not certify all CLI modes or physical model validity.

## Cost and failures

One saved candidate and one compiler/JIT process per target model; no observed target repair. Historical CUDA.jl attempts unknown. Reference P1 mapping correction is separate from target translation; scalar aliases had equivalent mathematical outputs. Exact consumed tokens unavailable; final-source proxies are partial, never total costs.

Reproduce: `python3 tools/verify_portable_julia.py --config src/silu-julia/study_config_ka.json`, then JACC config reusing saved references. Do not run a new series merely to refresh metadata.

## Taxonomy adjudication

Historical forward2/backward3 scalar aliases compute the same mathematical outputs as vector variants. The defect is benchmark variant/mapping fidelity, not wrong scientific computation. The earlier S1 label is superseded by P1; its event ID remains an alias for provenance. No source or numerical result changed.
