# Sheath observations (RQ1–RQ5)

**RQ1:** All four versions pass the complete declared numerical contract for three named cases, with full raw particle/field trajectories and independent deposition, Poisson and leapfrog witnesses. The original500k case has2 million exact initialized doubles verified against original RNG helpers. The strict initial thermal-scale potential criterion rejected native repeats; a separately named, input-derived FP32/Poisson precision enclosure revises only that case's potential normalization. It is conservative (5.398V maximum cross-model bound versus observed0.000667V), and is not plasma-accuracy certification. Raw physical fields remain available.

**RQ2:** Each target's first GPU program/3 device bodies passed. CUDA.jl had one standalone host-parser repair before GPU submission. Actual context-inclusive target windows: {'cuda_jl': 5046177, 'ka': 3072494, 'jacc': 2704402}. CUDA.jl overlaps shared completion repair and is excluded from clean cost calibration; nested verification is not additive. Native criterion/measurement repairs are not framework failures.

**RQ3:** Mean trials2–5 seconds: {'cuda': 0.1622422762500015, 'cuda_jl': 0.16596079300000544, 'ka': 0.17405038125000005, 'jacc': 0.19727017575000377}. This sums original-size hot1-step, smaller hot25-step, and cold-neutral1000-step loops, including own Julia CPU field solves/transfers/diagnostics. No full hot500kx1000 or GPU-only runtime claim.

**RQ4:** Particle/grid atomic deposition, mixed FP32/FP64 computation, AoS24-byte particles, CPU SOR and context handoffs explain the relevant source structure; no counters establish a bottleneck. All three authored-device implementations use public framework APIs.

**RQ5:** These recorded APIs can express the source operators on CUDA, while numerical precision and host timing require independent review across every abstraction. No universal backend/performance/physical-accuracy recommendation follows.
