# Projectile study

RQ1–RQ5 / E1–E5. Configuration projectile_deterministic_n1048576_r10_v1.
Unchanged native CalculateRange kernel and original CUDA.jl kernel are used with a named deterministic fixture: 1,048,576 Float32 angle/velocity pairs, ten launches per trial, block256. This is not the native default ten-million-element RNG run. Native hardcoded upper bound remains unchanged; exact multiple-of256 launch avoids excess threads. Original CUDA.jl main.jl and Makefile are preserved byte for byte in the child folder.

The original height expression `(v*v*s*s)/2*g` is preserved, including its reference semantics. Inputs and all 3,145,728 range/time/height outputs are shared and checked in all five trials. Native original CPU verifier and independent Julia scalar oracle accept finite values at absolute tolerance1, relative0. Cross-model binary comparisons show exact equality, maxerror0 for both variants against both references; CPU/GPU transcendental rounding can be nonzero but satisfies original threshold.

Timers cover synchronized host kernel-launch loops divided by ten. Input reset, transfers and device JIT precede timing. Trial1 excluded; mean trials2–5 is reported below. Native uses original five-float AoS while all Julia versions retain original SoA; native also stores inputfields. Therefore performance differences cannot be attributed solely to the abstraction. No measured roofline or causal bottleneck claim is made. Ontology independent-map/transcendental motifs reviewed; automated YAML must not substitute for counters.

| Model | Five times (us) | Mean retained (us) |
|---|---|---|
|cuda|107.463000, 108.180000, 107.634900, 107.981900, 107.586900|107.845925|
|cuda_jl|5283.920800, 38.652300, 37.208300, 36.951300, 36.964300|37.444050|
|ka|5219.052300, 54.789400, 40.857400, 40.791300, 42.238400|44.669125|
|jacc|2952.422000, 62.163500, 61.816500, 61.867600, 62.533500|62.095275|

One candidate, one evaluated compiler/device-JIT invocation and no compiler feedback repairs per new abstraction. JACC authored after KA verification. Reference/KA five-trial bundle reused unchanged, with explicit legacy adapter audit; no additional measurement series. Exact translation usage unavailable; saved-summary text proxy is partial and does not estimate total consumedtokens. Original CUDA.jl cost unchanged. No unsupported feature observed. Raw evidence: `.porting-state/portable_julia/runs/20261016T224431065977Z_projectile`.
