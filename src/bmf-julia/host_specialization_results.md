# BMF host function specialization control

The primary JACC source and all primary measurements remain unchanged. A separately declared diagnostic changes only `jlaunch(f,...)` to `jlaunch(f::F,...) where {F}`; two include paths change because the diagnostic source lives in the parent folder. No device body, launch geometry, input, acceptance rule, or timed region changes. All five diagnostic trials match the complete native and CUDA.jl captures exactly.

| Block | Five times (ms) | Required mean of last four (ms) | Median of retained trials 2–5 (ms) |
|---|---|---:|---:|
| Primary | 18.117, 18.639, 18.068, 17.277, 533.762 | 146.936 | 18.353 |
| Explicit F specialization | 17.182, 17.151, 16.853, 17.239, 17.387 | 17.157 | 17.195 |

The primary spike is retained under the required protocol. The diagnostic block has no analogous spike. A single chronological, unpaired block cannot establish that specialization explains the spike or the apparent difference between frameworks. CPU allocation/GC/dispatch profiling and counterbalanced repetitions would be needed. Trial 1 timing is excluded from both comparative statistics. The retained-trial median does not replace the required last-four mean. Diagnostic captures, a predeclared normalized source diff, and separate tuning-token boundaries are preserved. This configuration has role `diagnostic_tuning` and is excluded from the primary cohort and ordinary translation-cost calibration.
