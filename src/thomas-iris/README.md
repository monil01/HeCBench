IRIS DMem Thomas batch solve. Original argument order, synthetic matrix generator,
transposition, CPU repeated solve and Maximum error summary are preserved.
Default RUN_ARGS: 1024 16384 64 100. Both U and RHS use iris_rw so repeated
iterations operate on device results; only final RHS is flushed for verification.
Inputs L/D remain iris_r. Native and IRIS maximum errors are compared with
absolute tolerance 1e-12, with both required below 1e-9 in local validation.
Nonfinite solutions produce an error exit. Bounded validation includes batch
sizes not divisible by the block size and repeated solves; full default
performance workload is preserved but need not be used for bounded validation.
