# zoom: scientific and implementation observation

Zoom makes scientific operator and implementation mapping separable: whole-pixel interval means are not bilinear or overlap-weighted resampling, and output dimensions remain fixed through crop/pad. A geometry-based unit-cell intersection oracle computes exact rational means at actual Float32 endpoints, independently of the shared implementation and native CPU loops. Constant fields remain exactly one in every measured output, and all edge/crop pixels are checked. Source-backed tile capacities and pipeline stage count provide analysis traits; no causal shared-memory bottleneck claim follows from small-case timings.

## Reference audit

Historical CUDA.jl genuinely computes native box-span means/crop/pad, but omits the native shared staging tiles and block barrier, reading input directly for each output. Its CPU oracle uses Float64 ratio while device/native use Float32; edge kernel redundantly rewrites interior values. Study preserves original bytes, restores shared staging and original16x16mapping, aligns Float32 host coordinate arithmetic, and uses an algebraically equivalent outside-only clamp for native edge copies. These are mapping/oracle-precision changes, not an established historical wrong operator.

## Numerical and performance evidence

Configuration `zoom_original_bothfactors_sharedtiles_two_shapes_twofields_v1`, result `/home/imo/HeCBench_julia/.porting-state/portable_julia/runs/20261018T012019744608Z_zoom/result.json`. Five in-process trials per model; full outputs checked in every trial against both references; mean trials2–5. Warmup precedes trial1. Native kernel unchanged. This named configuration does not certify all CLI modes or physical model validity.

## Cost and failures

One saved candidate and one program/JIT check per Julia model, each with three authored kernel definitions, no observed target feedback repair. CUDAjl mapping/oracle-precision restoration recorded separately from target translation. KA capacityVal types imply multiple kernel specializations within one compiler-validation process, not extra translation attempts. Usage windows include verification and same-benchmark metadata. Historical CUDA.jl attempts unknown. Reference correction or documented P1 mapping phase is separate from target translation. Observed prospective usage intervals are recorded separately; exclusive billed translation cost is unknown. Final-source proxies remain partial.

Reproduce: `python3 tools/verify_portable_julia.py --config src/zoom-julia/study_config_ka.json`, then JACC config reusing saved references. Do not run a new series merely to refresh metadata.

## Image operator, parameters and memory evidence

The output retains original input dimensions. Expansion performs centered cropping; shrinking places the zoomed field centrally then replicates nearest active edges. Native factor pairs are retained with actual Float32 dimension products, ratios and coordinate endpoints. Whole intersecting source pixels receive equal weights; this is neither bilinear interpolation nor exact fractional-area resampling. Odd widths/heights and asymmetric crop/pad offsets are included.

All16,200 output pixels and16,328 exact input/control values are checked per capture. The independent unit-cell intersection witness computes rational unweighted means, direct crop/clamp geometry and roundedFloat32 expected values; all20captures pass. Constant-one fields remain exactlyone. Every source tile footprint fits the exact allocated96/551/560Float32slots; maximum measuredfootprint532slots, no missinghalo data. This does not certify huge-shrink fallback cases.

Native kernels/headers are unchanged. Historical CUDA.jl directglobal mapping was restored to source shared-tile dataflow and its Float64 CPUratio aligned to actualFloat32. CUDAjl/JACC shared capacities are dynamic; KA uses static Val-capacity specialization. Eachmodel authors3kerneldefinitions and has1candidate/programvalidation process, no observed feedbackrepair. JACC copies the complete template and adds an initialbarrier; edge kernel explicitly uses0sharedbytes. Nativefallback repeats blocks[0], so listed smaller blocks are not actually tried; unmeasured source-host limitation, not framework unsupported.

`per_component_performance.json` preserves8component×5series and means trials2–5. Pipeline ROI includes outputreset and edgepad kernel when shrinking, plus hostlaunch/completion overhead. The extra pipeline stage and larger sharedcapacity in shrink mode are confounds; no causal hardware bottleneck conclusion without counters. Usage windows include same-benchmark verification/metadata activity.


Prospective rawusage intervals include cached/repeatedcontext and coordination; the parentworkflow contains its childphases and must not be summed with them. Exclusive billed translation cost remains unknown; source-size proxies remain separately labeled.
