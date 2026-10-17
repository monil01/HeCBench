# Lomb–Scargle study

## Scientific meaning and independent oracle

The operator computes a scaled sinusoidal least-squares power over angular frequencies: P(ω) = 0.5 × scale × bᵀG⁻¹b, where G is the two-column cosine/sine Gram matrix. The native implementation diagonalizes those columns with an atan2-derived time shift. The independent Float64 oracle solves the unrotated 2×2 Gram system instead. The least-squares interpretation is described in [VanderPlas (2018)](https://doi.org/10.3847/1538-4365/aab766), also available as an [open preprint](https://arxiv.org/abs/1703.09824).

The fixture retains the original 1,000 uniform sample times, 100,000 frequencies, and noise-free signal of amplitude 2, angular frequency 1 and phase 1.57. Frequency means radians per input time unit, not cycles per unit. It does not evaluate irregular sampling, an astronomical dataset, weights, noise significance, or an offset/floating-mean model. The source's fixed Float32 scale 2/1.5 is preserved. Despite a normalize=True comment, it is not a data-energy normalization: powers near 1,332 are expected and are not restricted to [0, 1].

Every trial checks all 100,000 Float32 powers against the independent Float64 oracle and both GPU references with the original absolute tolerance 0.1. Maximum Float64 errors are approximately 0.00365. The original Float32 CPU reference also passes. All powers must be finite and nonnegative, the detected peak must lie within 0.002 of angular frequency 1, and power must respect the least-squares projection energy bound within tolerance. Saved artifact audits retain relative output L2 error, peak frequency/power, and hashes for every model/trial. Relative L2 accuracy is distinct from physical power normalization; off-resonance relative errors can behave differently from global errors.

## Mapping and measurement scope

Native source selects 100,096 blocks of 256 threads, rather than 391 blocks sufficient for the 100,000 frequencies. All four versions preserve this policy: 25,624,576 launched threads, 391 blocks containing valid work and 99,705 idle blocks. This is a source-policy observation, not an inferred bottleneck or a launch optimization. Each valid lane executes a serial loop over 1,000 samples. CUDA.jl uses its grid dimension directly; KA/JACC receive the equivalent source stride. JACC explicitly requests zero shared memory.

Frequency lanes reuse the same sample index at each loop iteration. Native inputs have const/restrict declarations; actual Julia read-only inference, cache instructions and generated memory traffic are unknown. Sincos and atan2 are transcendental operations and must not be counted as one ordinary FLOP without a stated convention. Registers, spills, SFU utilization, occupancy and bottleneck counters were not measured here. The sub-MiB working footprint and source launch policy limit scaling and bandwidth conclusions.

All four use explicit warmup, five synchronized in-process trials, mean of the last four, and ten overwrites per trial. Input/oracle generation and loading, allocations, resets, transfers and all checks are outside timing. The essential binary fixture and Double oracle have a reproducible C++ generator, wrapper, flags and hashes; different host libm/toolchains may generate different bytes, so the measured saved fixture is authoritative.

KA and JACC each passed one candidate, one check and one kernel with zero repairs. The old Julia touch/dummy PASS is archived; the current reference repair, native harness and independent oracle are separately costed. Historical generation effort is unknown. KA closed before the JACC draft. Consumed usage includes context, cached context and coordination; broad workflow and child windows overlap and are not additive or monetary billing.
