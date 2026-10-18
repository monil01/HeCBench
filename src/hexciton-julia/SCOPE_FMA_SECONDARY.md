# Revised secondary verification contract

This configuration was declared after the original native strict-zero auxiliary check rejected a capture. Native and authored GPU algorithms are unchanged. All four models receive fresh five-trial series; the original CUDA.jl strict-zero pass remains historical evidence and is not reused here.

The identity-only auxiliary check now uses the exact componentwise set `{0,+r,-r}` derived in `identity_FMA_secondary_contract.json`; its hash is bound in the new configuration before any new GPU execution. The original full-output numerical tolerances, all other scientific checks, input bytes, canaries, source 1001 repeats and 75-case timing scope are unchanged. This is a post-observation checker revision, not a native scientific defect or framework failure. It is excluded from untouched-native aggregates. See original SCOPE.md for the measured real-H and full-package limitations.
