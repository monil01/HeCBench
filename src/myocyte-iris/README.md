Myocyte is deferred after an actual native HIP versus IRIS HIP numerical artifact mismatch.
No validated executable port is supplied here.

Original CUDA and HIP CAM kernels write communication values to d_finavalu[com_offset], racing ECC derivative writes 0..2; d_com is never written but copied to host and consumed by kernel_fin. Changing the destination, initialization, or synchronization would change required source semantics.

The first DMem translation preserved the equation bodies, adaptive solver,
integer device time conversion and two-block geometry. It built and ran on
MI100, but returned STATUS -1 and did not match native HIP at interval 1.
Repeated runs reproduced each implementation's own result, without establishing
correctness across different uninitialized allocations.

Failed translation sources, logs and trajectories remain in
.porting-state/porting_logs/myocyte-iris/attempts/attempt_01. No final_correct
snapshot is claimed. Original source and datasets are untouched. Defaults:
interval 100, workload 1, original output.txt format; no random number usage.
