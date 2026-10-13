IRIS DMem spline value/gradient/Hessian port. Default compile-time sizes remain
12000 walkers and 2003 splines; all initial coefficients, CPU reference and
atol2 checks are retained. `HECBENCH_WSIZE` and `HECBENCH_NSIZE` allow bounded
validation builds with identical dimensions in an isolated HIP baseline.
Use `make clean` before changing `EXTRA_CFLAGS` size overrides.

Validated 8 walkers / 67 splines, including the nonmultiple-of-256 tail.
Host polynomial arrays explicitly update DMem per walker; output arrays use
iris_rw to retain prior walkers. Inverse grid factors convert to float exactly
as native kernel launches do. Outputs flush before reference verification.
Failed scalar conversion and repair are recorded in local attempt history.
