# Logic rewrite: RQ observations

- **RQ1:** Full source 15-pass pipeline on four structured K3 AIGs, all 32 outputs exhaustively checked over 16/64/512/4096 assignments. All20 captures, each169762 exact fields and5659 input fields, match; independent topology/truth/continuity/fanout replay passes. Original256-input default is not exhaustively certified.
- **RQ2:** CUDA.jl required29 math-generator processes and25 subsystem device-JIT processes before one successful complete candidate. KA/JACC each passed one complete candidate reusing the accepted math. Actual scoped counters and context in effort_summary.json; first complete pass is not first-try translation. Compiled exponentiation replacing XOR was rejected numerically.
- **RQ3:** Mean seconds trials2–5: native4.1628664985, CUDA.jl4.15935645225, KA4.162142544, JACC4.1585666155. Common CPU/native Thrust work insideROI; staging/readbacks/oracles outside. Not isolated-kernel/defaultCLI timing.
- **RQ4:** CUDA pointer/POD/device-heap/warp extensions, source geometry, generic framework dispatch, adapted stable sorting/storage and native Thrust host interoperation are explicit covariates. No measured bottleneck or instruction-equivalence claim.
- **RQ5:** Named-pipeline exact correctness supports qualified interoperability, without backend-independent portability or intrinsic framework/token ranking.
