# CED observations

Configuration `ced_defined_zero_borders_12resident_frames_full4GPUstages_3sizes_aligned_H0_v2`. Conditional native reference; original dataset CLI is unverified.

## RQ1: fidelity

All20 complete artifacts match bit for bit: 803,730 Int32 values each, including 401,922 actual input/parameter/guard prefix values. Independent coordinate CPU checks and four separate complete untimed stage traces have zero observed error. Original parser ASAN failure and 356 uninitialized-border reads are retained; defined borders pass initcheck. The source point threshold is preserved as a method choice, rather than replaced by connectivity hysteresis.

## RQ2: feedback and tokens

Each target passes its first authored candidate and actual device-JIT process. Four algorithm kernel definitions per target. Shared native preparation, CPU-audit tooling failures and annotation are separate; actual usage windows include context/cache and inherited helper reuse.

## RQ3: execution

| Version | Mean of trials 2–5 (µs) |
|---|---:|
| cuda | 190.062 |
| cuda_jl | 287.548 |
| ka | 330.403 |
| jacc | 662.341 |

Resident12-frame/48-launch host-elapsed ROI with whole-device completion. No transfers, allocation, original thread partitioning or image IO inside. Working buffers total401,808B. No kernel-only or causal abstraction ranking.

## RQ4: mechanisms and RQ5: selection

All three versions use their own framework device APIs. KA local memory is static; the other tiles are dynamic. JACC adds a324-value template preload/barrier, followed by complete source-image overwrite; actual initial/final scratch values are zero. Native constant symbols become global read-only-by-code coefficient arrays. Threshold scratch H0 was selected prospectively and symmetrically, with originalH1296 evidence retained. These are measured configuration/resource covariates, not unsupported features or profiler-derived bottlenecks.

Synthetic image/stage fidelity is not full canonical Canny or application-quality validation. See JSON for source hashes, complete limits and usage windows.
