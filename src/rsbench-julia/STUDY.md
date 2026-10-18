# RSBench: source resonance cross-section restoration

All four models pass all five complete captures: 46840 Float64 words per trial, 16149-word actual input prefix, all 1608 timed argmax outputs plus four cross-section components per lookup from separate untimed replay diagnostics. Independent 70-digit mpmath source-formula and sequential bigint LCG witnesses check every value. RNG/argmax/readonly/70 typed guard fields require exact bytes; macro XS uses predeclared abs1e-9+rel1e-8. Complete saved-capture audit repeats these distinct checks; global float tolerance cannot admit a guard/index discrepancy. Diagnostic values are replay states, not extra original timed stores.

## Scientific scope

Nuclear/neutron cross-section lookup proxy, source HMsmall68 nuclides/12 materials/seed42, avg12 poles/3 windows, eight N1/33/257/513 both0K/Doppler cases. Timed source uses skip-ahead UInt64 RNG, weighted FP64 complex resonances, firstmaximum component; native source lookup unchanged. Original large production CLI and physical reactor prediction/transport quality remain unverified.

Source window initializer end=ctr+space-1 versus exclusive loop bound leaves endpoint poles unvisited. Complete coverage and independent nonzero omitted-endpoint contribution witnesses retained. This source convention is preserved; it does not alone prove an external physics contract requires a correction. Source fast_exp approximation differs from mathematical exp by up to7.8% on exercised Abrarov arguments, not a cross-section physical error bound. Independent oracle checks the implemented approximate formulas, not exact physical Faddeeva. Its111/34736 branch counts are distinct CPU witness evaluations, not observed GPU performance counters.

## Mapping and performance

Historical Julia was S1 unrelated touch/LCG kernel with hardcoded checksum/model output. Restored full genuine source algorithm in shared immutable complex/scalar helpers; CUDA.jl, KA and JACC own actual dispatch. All framework_device_kernel, no vendor device fallback/backend extension. One authored entry with two compile-time Val specializations: timed argmax, untimed full arithmetic diagnostics. Source256-thread geometry and AoS Pole72B/Window32B field strides retained; dead padding unobserved. JACC explicit zero dynamic shared; no authored shared/atomic/barrier. Private tuple lowering/resources unprofiled. Five checked trials, mean2–5, warmup outside; timing sums eight source lookup kernels plus context completion only.

## Effort

Current reference one authored device candidate/one device JIT, two whole-program checks: shared generator-whitespace B2 before JIT. Initial weak CPU Meta.parseall check ignored error Expr; recursive syntax check fixed. Shared CPU mpmath environment missing B1 resolved pinned1.3.0; auxiliary endpoint exec namespace T1 fixed, no device failure. Native first build/process/five trials valid. KA and JACC each first candidate/process valid, reusing verified common restoration, not independent from-scratch costs. KA activity window also contains CPU endpoint-witness work and may undercount its final small namespace fix because END shared that tool round. Raw counters preserved. All windows include cached/repeated context, science/metadata/waiting/coordination; overlapping workflow is not summed or called exclusive billed coding. Historical cost unknown.

Final proof `.porting-state/portable_julia/runs/20261018T120034982422Z_rsbench/result.json`; canonical `study_config_jacc.json`; independent_capture_witness.json; per_case_performance.json. No source/timing edits after verified proof.
