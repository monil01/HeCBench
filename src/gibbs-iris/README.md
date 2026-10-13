# Gibbs IRIS
Retains input CSV argument (integer y and float n), trial argument/default1000,
Makefile run trials5000, seed123, Marsaglia gamma and host Metropolis sampler.
HIP RNG state layout and hiprand kernel functions match native HIP baseline;
CUDA RNG stream is backend-specific as in existing HIP sibling. DMem states are
initialized by setup and updated with iris_rw; theta remains a device intermediate.
Flat/log partial sums are flushed before host sampling. Missing original dataset
is not generated in source; validation uses explicitly documented bounded fixtures.
