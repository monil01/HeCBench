# vol2col: scientific and implementation observation

Volume lowering and folding admit an independent sparse-operator witness. Decoding each final column position gives its unique logical volume source or zero; counting those references predicts the complete folded result as coverage*x. This separates transpose composition from an inverse claim and makes stride/dilation sparsity explicit. Both genuine historical CUDA.jl kernels and original native kernels remain unchanged. Complete initialized physical allocation tails are checked separately from logical operator values. Allfive source kernel sizes and two named stride/dilation policies are measured, without claiming defaultlarge-volume throughput.

## Reference audit

Historical CUDA.jl genuinely implements both native grid-stride kernels and fully zero-initializes its CPU column reference. Native eval mallocs its CPU column reference without initializing the oversized unused allocation suffix, then memcmps the full allocation against zero-initialized GPU columns. Study initializes all physical buffers and preserves both original native kernels and both genuine CUDA.jl kernel bodies. This is a host-validator initialization scope, not evidence of a wrong native device operator.

## Numerical and performance evidence

Configuration `vol2col_all5kernelsizes_two_stride_dilation_policies_fullbuffers_v1`, result `/home/imo/HeCBench_julia/.porting-state/portable_julia/runs/20261018T010417721865Z_vol2col/result.json`. Five in-process trials per model; full outputs checked in every trial against both references; mean trials2–5. Warmup precedes trial1. Native kernel unchanged. This named configuration does not certify all CLI modes or physical model validity.

## Cost and failures

One program/device candidate per Julia model with two definitions; CUDAjl reuses original bodies. Shared native timing adapter initially queried deviceproperties insideROI, unlike original source; S2 preserved and corrected by hoisting blocks calculation, rerunning native only while unchanged Julia evidence reused. This is shared measurement effort, not a target compiler failure. KA required one B2 macro-expansion repair (remove redundant trailing returns), yielding two saved program candidates/checks but one process reaching actual device JIT; JACC adopts the validated form. KA window includes same-benchmark source/scientific metadata drafting during verification. Historical CUDA.jl attempts unknown. Reference mapping is preserved; this phase adds a measurement harness rather than a semantic repair. Observed prospective usage intervals are recorded separately; exclusive billed translation cost is unknown. Final-source proxies remain partial.

Reproduce: `python3 tools/verify_portable_julia.py --config src/vol2col-julia/study_config_ka.json`, then JACC config reusing saved references. Do not run a new series merely to refresh metadata.

## Mathematical and allocation scope

All five kernel sizes (1,3,5,7,9) are checked for the source stride/dilation2 policy and a named stride/dilation1 policy. Each capture contains949,020 physical output values and6,500 exact input/control values. The independent flat-column sparse-map witness checks every value and predicts folded output as coverage*x, including fully initialized unused tails. All twenty captures pass. Backward consumes forward-derived columns as in source; arbitrary independent column gradients and generic inverse claims are outside scope.

For stride/dilation2 only12 of210 logical voxels have positive coverage in this fixture; for stride/dilation1 all210 are covered for kernel3andlarger. Maximum overlap varies1..105. This is an explicit sampling operator, not evidence that the source reproduces standard convolution output dimensions for everykernel. Logical volume210 is stored contiguously; the remaining420 values are unused allocation tail, not a halo-strided image.

Both native device kernels and genuine historical CUDA.jl bodies are unchanged. Fully initialized CPU column tails resolve a source host-checker initialization limitation. The first native adapter timed a device-property query; its numerically valid series is preserved but timing is rejected. Only native was remeasured after hoisting query outsideROI; unchanged Julia evidence reused. KA trailing-return macro failure is B2 before deviceJIT, not unsupported algorithm.

`per_component_performance.json` preserves all20 five-trial component series and means of trials2–5. Wall-clock timing includes launch/completion overhead. No bandwidth, occupancy, arbitrary gradient precision or large-volume performance claim follows from these small dyadic cases. Usage records are full activity windows, not exclusive coding billing.


Prospective rawusage intervals include cached/repeatedcontext and coordination; the parentworkflow contains its childphases and must not be summed with them. Exclusive billed translation cost remains unknown; source-size proxies remain separately labeled.
