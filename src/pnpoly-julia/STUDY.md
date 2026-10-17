# pnpoly: scientific and implementation observation

The port retains every native point-tile width, not scalar aliases. A dyadic rectangle fixture provides an independent analytic membership witness and excludes boundary ambiguity. This contract verifies all configured outputs; it does not establish robust predicates for arbitrary or self-intersecting polygons. Optimized partial-tile inputs are excluded because native loads leave invalid private slots uninitialized.

## Reference audit

Historical CUDA.jl only executes a256-element touch kernel and prints a fixed checksum/PASS, not point-in-polygon. Genuine native crossing parity and all8tile mappings now implemented.

## Numerical and performance evidence

Configuration `pnpoly_rectangle600_n16384_all8tiles_r10_v1`, result `/home/imo/HeCBench_julia/.porting-state/portable_julia/runs/20261017T172746791712Z_pnpoly/result.json`. Five in-process trials per model; full outputs checked in every trial against both references; mean trials2–5. Warmup precedes trial1. Native kernel unchanged. This named configuration does not certify all CLI modes or physical model validity.

## Cost and failures

A shared host-fixture generator spacing error failed CUDA.jl and KA before device JIT. One measurement-helper repair, preserved separately from device candidates; corrected reference and KA second checks follow. JACC inherits corrected fixture. A subsequent device InvalidIRError exposed a captured index shared across base and tiled branches. Candidate2 separates the base index; this is recorded compiler feedback, not framework unsupported. Historical CUDA.jl attempts unknown. Reference correction or documented P1 mapping phase is separate from target translation. Exact consumed tokens unavailable; final-source proxies are partial, never total costs.

Reproduce: `python3 tools/verify_portable_julia.py --config src/pnpoly-julia/study_config_ka.json`, then JACC config reusing saved references. Do not run a new series merely to refresh metadata.
