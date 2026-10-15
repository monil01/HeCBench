# SpSM IRIS dmem port

Solves the same non-transposed CSR lower-triangular, non-unit-diagonal system A*C=B with alpha1.0. IRIS DMem owns all device arguments and output; HIP kernel dispatch occurs only through IRIS tasks.

P1 portability choice: the opaque cuSPARSE/hipSPARSE solve is lowered to a HIP CSR forward-substitution kernel with one independent RHS per thread and sequential rows. This preserves the mathematical operation, but not the vendor-library solver scheduling, summation order or performance. Native hipSPARSE is required only for validation, not for the IRIS port.

Original M/N/requested-nnz/repeat/verify argument order, seed123 and srand48(123), matrix input helpers, residual threshold0.01 and Makefile defaults (20082 128 150616 100 1) are retained. The native source's unsafe requested-nnz allocation is avoided by allocating CSR arrays after the returned actual nnz; this does not change generated inputs for valid source arguments.

Build make; run IRIS_ARCHS=HIP ./main M N nnz repeat verify. Optional HECBENCH_DUMP_PREFIX writes complete float32 outputs/expected original C for observational validation. CUDA/HIP siblings are untouched. Validation is bounded; the production default is not tested. No CUDA backend or vendor-library throughput parity is claimed.

The original NVIDIA disclaimer and U.S. Government End Users Notice are preserved in main.cpp.

Equivalent vendor-operation lowering was explicitly authorized for the remaining
IRIS coverage pass. The solver is a numerical port, not a hipSPARSE performance
replacement; validation compares every output against hipSPARSE and the original
known solution, rather than accepting timing output as correctness evidence.
