HMM IRIS DMem Viterbi retains defaults4096 states,4096 emissions,250 observations,
implicit libc RNG seed and exact complete CPU/GPU path equality (Success/Fail).
No command-line arguments were required in original benchmark. Prior/new probability
IRIS initialization follows random input generation because the runtime seeds libc
RNG; original implicit seed1 is preserved. Optional HMM_STATES/HMM_EMISSIONS/
HMM_OBSERVATIONS compile macros enable bounded validation with original defaults.
DMem buffers ping-pong each iteration, preserving original old=new update semantics
without host round trips. All outputs flush before traceback; intermediates iris_rw.
CUDA kernel unvalidated. Reference CPU remains separate verification, not fallback.
