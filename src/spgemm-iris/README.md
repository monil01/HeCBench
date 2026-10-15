# SPGEMM IRIS dmem port

This source benchmark actually computes sparse A times dense B into dense C, using both COO and CSR representations. Both paths are retained with alpha1/beta0/nontranspose and the original source input construction, unseeded drand48 sequence, seed123 shuffles, mandatory six arguments, repeat/verify semantics, threshold0.01 and two default run cases.

P1: opaque cuSPARSE/hipSPARSE SpMM is lowered to two raw HIP kernels, one output per thread. COO row lookup uses binary search on source-generated sorted tuples; CSR uses original offsets. Floating summation order/scheduling and performance differ from vendor implementations. No vendor throughput parity is claimed. Inputs for both paths are constructed before IRIS init to preserve source RNG sequence. Host device memory/dispatch uses IRIS DMem/tasks only.

Build make; run IRIS_ARCHS=HIP ./main M K N nnz repeat verify. Optional HECBENCH_DUMP_PREFIX exports full output/reference floats for evidence. Valid input nnz must lie in(0,M*K]; undefined source arguments are rejected. Native siblings are not modified. Bounded HIP tests do not validate the full production default suite. CUDA backend is not provided.

Original NVIDIA disclaimer and U.S. Government End Users Notice are preserved in main.cpp.

The remaining-port coverage pass explicitly authorizes these equivalent HIP
implementations of vendor operations. Validation compares complete COO and CSR
outputs to unchanged native hipSPARSE and the original dense CPU oracle.
