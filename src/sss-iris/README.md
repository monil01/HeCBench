SSS is deferred because its required GNU Scientific Library dependency is
unavailable. The original native HIP build fails with
`fatal error: 'gsl/gsl_integration.h' file not found`. System pkg-config,
loader library cache, and checked system/local/Conda include and library paths
do not provide GSL. The original DATA files are present and preserved.

The search uses GSL integration, special functions, and random distributions
throughout its CPU model/scoring logic. Replacing these with different
numerical algorithms would change the benchmark. A usable GSL installation
and native numerical baseline are required before validating an IRIS port.

No executable IRIS translation or GPU correctness claim is supplied. Original
CUDA/HIP files, tuning defaults, float/short scalar types, RANDOMSEED=314159265,
and the Makefile's GSL_RNG_SEED=123 invocation remain unchanged. This is an
external dependency deferral, not a HIP runtime or shell infrastructure error.
