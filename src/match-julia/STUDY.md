# match: scientific and implementation observation

The study translates all ten native matching mappings. Exact dyadic inputs permit an independent arbitrary-precision integer Gram-matrix oracle for complete scores and indices. Ties exercise native reduction priority: distinct optimizing variants can choose different valid maximizing indices, which is separated from mathematical maximum correctness. A zero query confirms score0/index-1. Shared tiling, padding, register blocking and prefetch are source-backed traits for scoped performance analysis; no causal bandwidth or occupancy claim is made without counters.

## Reference audit

Historical CUDA.jl uses GEMM plus row argmax for every MatchGPU1..10 label rather than the ten native optimization mappings. This may compute the same positive unique-maximum operator but does not preserve variant geometry, tiling, shared layout or reduction. Its supposed CPU reference is its own GPU output. Study restores ten actual kernels and independent integer-dot witnesses; no wrong scientific operator is inferred merely from the historical mapping substitution.

## Numerical and performance evidence

Configuration `match_all10_128points128dimensions_dyadic_and_zero_ties_v1`, result `/home/imo/HeCBench_julia/.porting-state/portable_julia/runs/20261018T004736764768Z_match/result.json`. Five in-process trials per model; full outputs checked in every trial against both references; mean trials2–5. Warmup precedes trial1. Native kernel unchanged. This named configuration does not certify all CLI modes or physical model validity.

## Cost and failures

CUDA.jl one saved candidate/program check compiling ten definitions. KA three candidates/checks, two B4 compiler repairs: distinct Cartesian/linear index variables and explicit tuple replacing closure across barriers. Shared Python oracle B2 syntax and B1 absent-numpy were resolved with standard-library integer arithmetic. A metadata helper syntax attempt is retained separately, not a device candidate. Historical CUDA.jl attempts unknown. Reference correction or documented P1 mapping phase is separate from target translation. Observed prospective usage intervals are recorded separately; exclusive billed translation cost is unknown. Final-source proxies remain partial.

Reproduce: `python3 tools/verify_portable_julia.py --config src/match-julia/study_config_ka.json`, then JACC config reusing saved references. Do not run a new series merely to refresh metadata.

## Complete variant and numerical scope

All ten native variants, including the final streamed-prefetch implementation, are measured separately in `per_component_performance.json`. Five captures per component discard the first and average the last four. The reported aggregate sums 20 component means (two fixtures), rather than treating ten labels as one kernel. All 5,120 scores/indices and 65,542 input/control values are checked per capture. The independent arbitrary-precision integer dot-matrix witness validates all twenty captures.

All native kernel bodies are byte-preserved in a source snapshot with only an NPTS macro guard added. NPTS128 is an explicit study parameter; NDIM128, native block geometries, and shared byte capacities are retained. Ordered scalar components/private tuples replace C++ float4 syntax; actual machine vectorization/register pressure is not profiled. JACC initializes the full shared template and adds one initial barrier in variants2–10; variant1 explicitly allocates zero shared bytes.

Every first-fixture query has positive argmax ties. Variants1–5 use earliest-global tie priority; variants6–10 use grouped priorities. Different native variants may return different valid maximizing indices. Both references and each port are compared for the same variant, and the independent oracle enforces its actual priority. Zero queries return the native score0/index-1 sentinel. These are not unrestricted signed argmax or arbitrary near-tie FP32 accuracy claims.

KA usage includes independent-witness and same-benchmark metadata work while verification queued; JACC includes metadata preparation. Raw usage remains activity-window cost, not exclusive translation billing.


Prospective rawusage intervals include cached/repeatedcontext and coordination; the parentworkflow contains its childphases and must not be summed with them. Exclusive billed translation cost remains unknown; source-size proxies remain separately labeled.
