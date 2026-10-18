# degrid: scientific and implementation observation

The operator is complex resampling from a regular grid at irregular coordinates, with a subpixel convolution table. A delta-kernel fixture provides an independent image-sampling identity, while scaled-integer arithmetic gives exact results for the general dyadic complex fixture. Both cover every subpixel bin, borders and points outside the logical image; the full padded storage and physical load range are checked. Native and CUDA.jl use warp shuffles. KA and JACC use a source-documented shared-memory emulation of that reduction dataflow, adding 4KB and block barriers. This is a selected portable mapping, not proof that either framework lacks all alternative shuffle implementations. Its timing cannot isolate the cost of the abstraction from the chosen reduction mapping without a controlled alternative and profiling.

## Reference audit

Historical CUDA.jl already implements the complex convolution and native warp reduction. It uses Float64 floor/bins instead of native floorf, guards image loads before bounds rather than using padded source storage, and sorts by kernel sub-bin order rather than the native swapped-axis locality key. These are precision/input/memory mapping differences, not evidence of a wrong configured scientific operator. Named adapter aligns Float32 floorf and fully initialized padded storage.

## Numerical and performance evidence

Configuration `degrid_fp64_image64_gcf32_grid8_points64_dyadic_and_delta_v1`, result `/home/imo/HeCBench_julia/.porting-state/portable_julia/runs/20261018T001218501526Z_degrid/result.json`. Five in-process trials per model; full outputs checked in every trial against both references; mean trials2–5. Warmup precedes trial1. Native kernel unchanged. This named configuration does not certify all CLI modes or physical model validity.

## Cost and failures

One candidate and one complete compiler/JIT validation process per Julia model, with no observed compiler, runtime or numerical feedback repair. The historical CUDA.jl phase aligns precision and measurement inputs; no wrong-scientific-operator repair is claimed. Static mixed-precision binning and point-order scope findings are preserved separately from measured numerical outcomes. Historical CUDA.jl attempts unknown. Reference correction or documented P1 mapping phase is separate from target translation. Observed prospective usage intervals are recorded separately; exclusive billed translation cost is unknown. Final-source proxies remain partial.

Reproduce: `python3 tools/verify_portable_julia.py --config src/degrid-julia/study_config_ka.json`, then JACC config reusing saved references. Do not run a new series merely to refresh metadata.

## Exact convolution and input scope

All 20 captures pass an independent scaled-integer complex convolution oracle exactly. Products have denominator 1024 and their integer numerators fit Int64; both complex components are checked. The second fixture uses a delta convolution kernel, so each result independently equals the image at the floored coordinate or zero outside it. The two cases each cover all 64 subpixel bins. Every padded image entry is serialized, its zero initialization is checked, and every source padded-load address remains in allocated storage. The original FP64 tolerance of 1e-7 remains unchanged.

The source stores coordinates in FP64 but calls Float32 floorf. For an off-lattice coordinate x=5−2^-28, source arithmetic gives main_x=5 and sub_x=-1, while Float64 floor gives main_x=4 and sub_x=7. This is a static source-arithmetic limitation, not a measured GPU failure. All measured coordinates are exact Float32 dyadics, so the issue is inactive. A separately identified off-lattice configuration remains unmeasured; no general FP64-coordinate or full-CLI certificate is claimed.

The named point comparator uses floor fractions and a swapped-axis sub-bin key. The study_common source comment about retaining native qsort order applies to nonnegative points: original qsort uses modf, so its ordering differs for the deliberately included negative coordinates. All four measured models consume the same exact ordered point buffer. This shared fixture choice changes locality from a hypothetical original negative-point CLI run, without changing the defined convolution or introducing a translation-specific input difference.

## Independent per-model mapping

CUDA.jl uses its own exported warp shuffle API and remains a framework device kernel with a CUDA-specific warp feature. KA and JACC use genuine generic shared-memory reduction code: 4096 shared bytes and 44 algorithm block barriers per block. JACC also copies the initialized template and has one initial block barrier. No CUDA device shuffle calls or fallback kernels are hidden inside those variants. The emulation preserves every lane's reduction dataflow, including out-of-range shuffle behavior. This is a selected portable mapping, not evidence that all framework-specific shuffle alternatives are unavailable. A controlled hybrid alternative and profiling would be needed to attribute the timing gap causally.

One candidate and one compiler/JIT validation per model passed, with no observed target repair. Historical CUDA.jl precision/input/padding alignment is P1 mapping work, not a demonstrated wrong scientific operator.


Prospective rawusage intervals include cached/repeatedcontext and coordination; the parentworkflow contains its childphases and must not be summed with them. Exclusive billed translation cost remains unknown; source-size proxies remain separately labeled.

The observed JACC activity interval also includes the coordinator-requested transfer bookkeeping for untouched tsa and winograd. No audit, draft or GPU work for those cases was performed. Raw counters remain unchanged and are not exclusive coding cost.
