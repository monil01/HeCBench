# dwconv1d: scientific and implementation observation

The independent derivative witness checks the mathematical reverse operator, beyond reproducing forward values. Historical forward-only correctness could not certify backward paths; hardcoded text was not evidence. Full native perbatch weight gradients and input gradients now checked, with exact adjoint dot-product identity on dyadic fixtures. Named timings measure only the original two kernels, not PyTorch loss/backprop or reduced weight-gradient pipeline. JACC public shared preload and Julia scalar accesses are explicit mapping differences, so speed ratios cannot be read as intrinsic abstraction penalties.

## Reference audit

Historical Julia forward implements genuine causal convolution but omits native backward gradients and prints hardcoded backward-correctness/profiler measurements. Current study restores actual forward/backward work and full derivative checks. Forward scalar grid mapping also aligned to source vector4/grouped shared-memory work. Native timex.cu and PyTorch wrapper unchanged; no native scientific repair.

## Numerical and performance evidence

Configuration `dwconv1d_source_grouped_forward_backward_3fixtures_v1`, result `/home/imo/HeCBench_julia/.porting-state/portable_julia/runs/20261018T052958761424Z_dwconv1d/result.json`. Five in-process trials per model; full outputs checked in every trial against both references; mean trials2–5. Warmup precedes trial1. Native kernel unchanged. This named configuration does not certify all CLI modes or physical model validity.

## Cost and failures

Two reference whole-program checks: first shared frontend B2 before device JIT, then pass; authored device candidate/process count1. KA first candidate/process pass. JACC prospective result below. Reference phase includes independent oracle/native adapter; KA window also includes same-benchmark host adjoint witness annotation. Historical true forward effort and fake backward creation effort unknown. Historical CUDA.jl attempts unknown. Reference correction or documented P1 mapping phase is separate from target translation. Prospective harness activity counters retained including cached/repeated context and coordination; nonexclusive, historical usage unknown and source proxies partial.

Reproduce: `python3 tools/verify_portable_julia.py --config src/dwconv1d-julia/study_config_ka.json`, then JACC config reusing saved references. Do not run a new series merely to refresh metadata.

## Scientific and mapping qualification

All20artifacts bitidentical:19679exactinputprefix+28032fullforward/perbatchgradW/gradKvalues+144guards=47855Float32. Independent direct convolution and symbolic derivative errors actuallyzero on dyadic fixtures, though predeclared tolerance2e-5+2e-5abs unchanged. Independent exact bilinear adjoint identity holds all3cases after host batchsum of weight derivative, not a timed GPUreduction. Own frameworks allocate shared27648Bforward/15360Bbackward; JACC explicitly allocates these sizes but publicshared template preload adds work. Four scalar authored accesses perlane replace native float4 syntax; no emitted instruction equivalence claim. Named two-kernel ROI excludes original PyTorch allocation/loss/tanh/autograd/batchsum and full32x768x768CLI. Meansseconds {"cuda": 0.00014720375, "cuda_jl": 0.00082605, "ka": 0.0008939880000000001, "jacc": 0.0015994240000000001}. Reference two wholeprogram checks include one shared parsefailure beforedeviceJIT; one authoreddevicecandidate/process eachmodel, no actual devicecompile repair.
