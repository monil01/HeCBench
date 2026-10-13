# Integer division IRIS
Preserves repeat argument, original functional 100000 divisor/1000000 dividend
sweep and throughput/latency kernels, 100 warmups, default 100 repeats.
Functional buffer uses iris_rw and DMem.update(host) for every zero reset.
IDIVIDE_VALIDATION_DIVISORS, IDIVIDE_VALIDATION_DIVIDENTS and
IDIVIDE_VALIDATION_GRIDS optionally bound validation; unset retains original sizes.
FUNCTIONAL PASS explicitly reports completion of original zero-error check.
