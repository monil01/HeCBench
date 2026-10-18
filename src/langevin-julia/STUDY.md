# Langevin function: fidelity and mathematical accuracy

This benchmark evaluates L(x)=coth(x)−1/x. It is a mathematical special-function microbenchmark, not stochastic Langevin dynamics. Source k0 uses cosh/sinh−1/x, k1 reciprocal tanh−1/x, and k2 an explicit Float32 FMA polynomial. Every original Julia function is byte-preserved; only the final entry call is replaced with a shared fixture/warmup/measurement harness. reference_preservation.json and reference_original bind that proof.

## Declared and observed contract

Two4096-element fixtures cover the source negative interval and alternating signed dyadic values. Allthree modes use256threads×16blocks, R20 per component and explicit warmup. Five checked in-process trials retain every sample; means use2–5. Allocation/transfers/checks are outsideROI. The total sums six per-kernel-loop means; component_timings.json preserves each case/mode mean.

The supplemental fixture is NOT paired sign-symmetric: positive odd eighths and negative even eighths span[-1.75,1.625]. The old config identifier and one descriptive limit incorrectly say symmetric. study_adjudications.json supplies hash-bound descriptive corrections for all three immutable bundles. Actual fixture bytes, sources, oracles, tolerance and timing never changed. No odd-symmetry check is claimed.

The dual contract was declared atol5e−4/rtol2e−6 before execution. Actual all20 full33040-value captures are bit-identical across CUDA/CUDA.jl/KA/JACC. Each contains8208 actual input-bit/parameter prefix exports,24576 complete outputs and256 device guards. Independent policy acceptance uses per-value max(2e−6,8*2⁻²³/|x|) for direct formulas and2e−7 for the polynomial. Every output is finite; allguards exact. Saved-output audit independently recomputes raw absolute/relative/budget metrics and verifies all source bindings. Relative policy error uses floor1e−30, explicitly distinct from relative true-function error.

## Independent mathematical quality

CPU Float32 libm and fmaf evaluate source policies; a60-decimal-digit exp-based coth−1/x supplies independent mathematical truth. The source printed seventh-degree Taylor approximation is retained as a separate approximation baseline, not ground truth. Mathematical_quality.json records all120 case/mode/trial/model comparisons.

For the source interval, k0 and k1 maximum true-function absolute error is9.4374023e−5, while polynomial k2 is3.1898684e−8. At x≈−.0004492998123, the direct formulas give−.000244140625 versus trueL≈−.0001497666021 (about63% relative error); k2 gives−.0001497665944. Every frontend reproduces this source-policy behavior. Agreement is therefore distinct from numerical-method accuracy. Finite nonzero arguments only are certified; zero/large arguments/nonfinite values and original nonmultiple-of256 tails are excluded.

## Ontology, performance and cost

Each model has three genuine framework_device_kernel implementations with ordinary framework hyperbolic/FMA operations and no foreign device API, shared memory or atomics. JACC explicitly uses shmem_size0. Host allocations/completion alone do not make this a hybrid pipeline. GPUresources/counters are unmeasured, so no roofline or causal frontend-overhead claim is made. Total mean microseconds: CUDA16.414388, CUDA.jl19.511513, KA32.506513, JACC54.029638; each total sums six component means at this small launch-sensitive size.

Reference and both targets each passed their first process/JIT check; targets each have one authored program candidate and three device definitions, no feedback repairs. Original CUDA.jl kernels were preserved, not repaired. Historical cost remains unknown; source proxies and context/cache/coordination-inclusive prospective usage windows are separate, nonexclusive measures. Four essential CPU binaries are reproduced by generate_oracle.py and bound in fixture_provenance.json with actual CPUlibm path/hash and Python version. Final study_config.json selects20261018T074747749116Z_langevin. Root alone commits each verified target; canonical guides and worker checkpoint resume the queue.
