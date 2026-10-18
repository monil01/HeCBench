# upsample: scientific and implementation observation

Upsampling is a linear replication operator; its backward kernel is the transpose, which gives an independent scientific witness beyond agreement between translations. The study checks every forward pixel and backward gradient at all18 source launch settings for two shapes, including odd dimensions, tails, multiple batches/channels. Integer-numerator arithmetic establishes <Ux,g>=<x,U*g> exactly for the finite dyadic fixtures. Native kernels and genuine historical CUDA.jl kernel bodies remain intact. A shortened legacy forward checker is replaced by exhaustive validation, without inferring a kernel failure from that checker limitation.

## Reference audit

Historical CUDA.jl genuinely implements all four source upsample kernels with corresponding block geometries. It already scalarizes native float2 stores/loads in the second mappings. Study preserves these four kernel bodies and adds identical named inputs, context-wide five-trial measurement, and full-output independent validation. Native and historical validators check only B*C*(H/2)*(W/2) forward elements instead of4*B*C*H*W, a checker blind spot rather than an observed device-kernel defect.

## Numerical and performance evidence

Configuration `upsample_all4kernels_all18launches_two_shapes_full_adjoint_v1`, result `/home/imo/HeCBench_julia/.porting-state/portable_julia/runs/20261018T005536114753Z_upsample/result.json`. Five in-process trials per model; full outputs checked in every trial against both references; mean trials2–5. Warmup precedes trial1. Native kernel unchanged. This named configuration does not certify all CLI modes or physical model validity.

## Cost and failures

CUDA.jl reuses four genuine historical definitions and adds a harness; target models author four definitions each. Program/JIT checks compile all relevant launch shapes before five measured trials. Prospective usage windows include verification and metadata activity, not exclusive coding cost. Historical CUDA.jl attempts unknown. Reference mapping is preserved; this phase adds a measurement harness rather than a semantic repair. Observed prospective usage intervals are recorded separately; exclusive billed translation cost is unknown. Final-source proxies remain partial.

Reproduce: `python3 tools/verify_portable_julia.py --config src/upsample-julia/study_config_ka.json`, then JACC config reusing saved references. Do not run a new series merely to refresh metadata.

## Complete launch and operator scope

All four kernels and all eighteen native launch settings are checked for both shapes. `per_component_performance.json` preserves all36 five-trial series and means of trials2–5; the aggregate sums those means. Each capture includes115,830 output values and12,880 exact input/control values. An independent output-coordinate pullback and transpose-scatter oracle proves <Ux,g>=<x,U*g> exactly in integer arithmetic. All twenty captures pass.

The shortened legacy forward check is a validation blind spot, not an observed kernel defect. Original native source and four genuine CUDA.jl kernel bodies are preserved. Four ordered scalar accesses replace native float2 syntax; actual generated vectorization is unprofiled. All block settings are retained, with no timing selection.

Useful logical work is0FP operations/20bytes per forward input and3FP additions/20bytes per backward input (nominal0.15FLOP/B). These are source-level logical quantities, not measuredDRAM traffic, compiler instruction counts or a proven bottleneck; integer indexing is excluded. Large tensors, negative zero, nonfinite values and extreme indices remain outside this named study. Usage windows include same-benchmark metadata and verification activity.


Prospective rawusage intervals include cached/repeatedcontext and coordination; the parentworkflow contains its childphases and must not be summed with them. Exclusive billed translation cost remains unknown; source-size proxies remain separately labeled.
