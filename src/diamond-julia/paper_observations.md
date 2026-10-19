# DIAMOND observations (RQ1–RQ5)

- **RQ1:** All four versions agree on all five complete primary input/mask/guard captures; actual supplemental posteriors match an independently expressed 51-state HMM. This is a conditional masking-component contract, not alignment quality. The original CLI passed an idempotence check after CPU masking; actual compact-matrix layout repair matters: the controlled wrong layout misses 7,464 mask bytes.
- **RQ2:** Each target has one authored candidate and one actual device-JIT process. The interrupted CUDA.jl pre-draft review is preserved separately from its new translation window; native bounds resolution is separate. Context/cache/session-inclusive counters are not exclusive billing.
- **RQ3:** Mean trials 2–5 (µs): native 317.806, CUDA.jl 1669.208, KA 1688.218, JACC 1706.443. Three resident masking launches plus whole-device completion; qualified reference excluded from untouched-native aggregates.
- **RQ4:** Private fixed arrays/MVectors and mixed F64/F32 recurrence are explicit. Native primary compilation reports 48 registers/960 B stack and no spill instructions; the untimed observer has separate resources. No profiler, roofline, occupancy or causal explanation of the timing gap is claimed.
- **RQ5:** All three frontends support this private-array algorithm on the tested CUDA backend. Correct host layout, genuine unmasked input, and categorical threshold checks are prerequisites; another backend and full DIAMOND integration/quality are unverified.

The terminal k=0 lookup is a static one-past private-array access, not an observed sanitizer fault or wrong-mask witness. Its explicit guard produces byte-identical full masks and posteriors in a fresh native series; all old evidence remains preserved.
