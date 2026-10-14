The previous pointer-alias blocker is resolved using whole DMem buffers and
explicit timestep/cashflow offsets in HIP wrappers. Original pricing, SVD,
regression, exercise and reduction device bodies remain. Default paths32K,
timesteps100, run1, put and option parameters, RNG and make runs100 remain.
The built-in binomial/European comparisons run outside the GPU lock. Numeric
validation compares the GPU price against current native HIP at explicit
tolerance; no CUDA compiler is needed for the AMD baseline.
Earlier deferred attempts remain in .porting-state/porting_logs/aop-iris.
