Deferred: vendor-library SpGEMM, not a verified IRIS port.

CUDA calls cuSPARSE workEstimation/compute/copy with library-owned descriptors,
dynamic scratch/output sizes and CSR pointer binding. Native HIP substitutes
hipSPARSE. There are no application kernels to launch through IRIS DMem.
The local IRIS headers/runtime expose no hipSPARSE SpGEMM descriptor adapter.
A valid port would need a benchmark-local IRIS HIP library adapter that obtains
DMem device allocations and preserves library descriptor/stream lifetime, or
an explicitly approved algorithm replacement. A native HIP host allocation
and direct library call would violate the requested IRIS DMem host model.

No main/kernel is supplied and this directory must not be counted as runnable
coverage. Source/baseline snapshots and command logs are recorded in
.porting-state/porting_logs/spmm-iris/manifest.json with status deferred (D1).
Original seven arguments, srand(123)/drand48 initialization, CSR outputs and
verification tolerance 1e-2 remain in the baseline; they have not been changed.
