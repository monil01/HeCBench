# goulash: scientific and implementation observation

This cell-gating proxy requires state persistence over repeated nonlinear updates. Native originalverification capturesonlywarmupstate, so apparent PASS fails tocertifytimedrecurrence. Studycomparesfinalcomplete state after10updates fromexactzero againstoriginalCPUrecurrence; namedvaryingvoltage tests morethanoriginalallzeroVm without changingdevicekernel. Polynomialandexp costs requirecounters beforebottleneckclaims.

## Reference audit

Historical Julia main prints hardcoded count/time/PASS and performs no gate computation. Native main verifies only a copy captured after warmup, not finaltimedoutput; studyharness preserveskernel and verifiesfinal10stepstate.

## Numerical and performance evidence

Configuration `gate_fp64_n4096_varyingVm_10steps_zero_reset_v2`, result `/home/imo/HeCBench_julia/.porting-state/portable_julia/runs/20261017T161841793517Z_goulash/result.json`. Five in-process trials per model; full outputs checked in every trial against both references; mean trials2–5. Warmup precedes trial1. Native kernel unchanged. This named configuration does not certify all CLI modes or physical model validity.

## Cost and failures

One unchanged device-kernel candidate permodel; two successful process checks each across initial warm-state fixture and corrected bit-exactzero fixture. First series remains non-primary because cross-host exp-derived initialbits were not established identical. Measurement contract correction separate from kerneltranslation repair. Historical CUDA.jl attempts unknown. Reference S1 semantic repair separate from target translation. Exact consumed tokens unavailable; final-source proxies are partial, never total costs.

Reproduce: `python3 tools/verify_portable_julia.py --config src/goulash-julia/study_config_ka.json`, then JACC config reusing saved references. Do not run a new series merely to refresh metadata.
