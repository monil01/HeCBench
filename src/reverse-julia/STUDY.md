# reverse: shared-memory permutation and launch overhead

RQ IDs: RQ1, RQ2, RQ3, RQ4, RQ5. Experiment IDs: E1, E2, E3, E4, E5.
Configuration: reverse_native_minstd_seed123_iterations1_v1. Protocol: five_trials_last_four_v1.
Evidence: `.porting-state/portable_julia/runs/20261016T223650315854Z_reverse/result.json`, config.json, timings.csv, all raw logs and binary artifacts.
Ontology: ontology/benchmarks/reverse.yaml, SHA256 `cf48d5c0dd348118c72a2876e42f803a4282824567f09afe5b8fb0de47dc8f4e`; existing automatically inferred labels retained.

## Numerical confirmation
All four models initialize Int32 values 0 through255. Native libstdc++ minstd_rand0 seed123 with uniform[100,9999] gives109 launches. KA and JACC each match all256 native CUDA and CUDA.jl output values exactly on allfive trials; zero absolute and relative tolerance. Allfive independent parity-oracle checks pass. Native source kernel is included unchanged. The historical CUDA.jl MersenneTwister count generator differed from native; the study driver aligns it and preserves original_main.jl and original_Makefile. Later iterations of the generator are supported but only iteration1 is measured here.

## Runtime (microseconds)
Each timing is total elapsed synchronized launch-loop time for109 launches, including host dispatch; transfers/JIT excluded. It is not isolated GPU kernel duration. Trial1 discarded, mean trials2–5.

| Model | Five trials (us) | Mean lastfour (us) |
|---|---|---|
| cuda | 270.132, 270.263, 273.472, 270.703, 270.273 | 271.178 |
| cuda_jl | 263.693, 289.512, 281.383, 267.723, 274.272 | 278.223 |
| ka | 519.534, 617.146, 514.984, 513.594, 491.604 | 534.332 |
| jacc | 935.068, 688.106, 651.975, 626.225, 629.616 | 648.981 |

## Observation and explanation
The small one-block permutation succeeds in both frameworks, including local memory and workgroup synchronization. KA uses static @localmem; JACC.shared copies the complete256-element vector to1024 bytes of dynamicshared storage and barriers internally. JACC LaunchSpec uses oneblock,256threads,sync=false, avoiding extra synchronization per launch. No feature gap was observed at these versions.

The measured synchronized launch loops are slower for KA/JACC in this configuration. Candidate explanation: host/framework dispatch and dynamic launch setup contribute substantially for tiny kernels. This is a hypothesis; counters/traces and launch-duration controlled experiments remain pending. Alternatives include launch-wrapper compilation/configuration differences and dynamic versus static shared memory. No measured roofline, traffic, occupancy, or causal bottleneck is claimed. Integer permutation should use launch latency/bandwidth rather than floating-point roofline.

## Attempts and cost
Each new framework has one source candidate and no compiler-feedback source repair. KA's first invocation encountered an infrastructure dependency-install race and missing runtime preference before device JIT, not a language feature limitation; subsequent device JIT succeeds. KA smoke and central runner invocations are separately logged; five timing trials are not five attempts. JACC's first candidate succeeds in the central runner. Translation token records are partial source-text estimates only and cannot establish total consumption or fair abstraction cost ranking. Infrastructure setup/reference alignment/paper work are separate phases.

## Paper follow-up
Use as a shared-memory + launch-dominated case (RQ4), not general performance ranking or recommendation. RQ1 answered for this exact configuration; RQ2 costs partial; RQ3 measured one small workload; RQ4 mechanism unresolved; RQ5 generalization unresolved. Profiling, scaling, representative multi-block inputs, publication snapshot export, and combined claim updates are coordinator work. No unsupported outcomes are suppressed.
