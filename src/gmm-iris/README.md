GMM uses IRIS dmem for all GPU arrays and tasks for every original kernel.
Host EM reductions and cluster-order reduction retain the CUDA/HIP algorithm.
Cluster pointer fields are flattened into task arguments and reconstructed
inside each device kernel. Host mirrors are not native device allocations.
Original CLI and 500-cluster default are retained. Input data must be supplied.
The original CPU reference verifies results; GPU execution requires IRIS_ARCHS=HIP.
