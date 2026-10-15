# Bray–Curtis IRIS DMem port

`make run` retains the original four forward workload shapes, CPU PyTorch
seed 0 tensors, float32/float64 types, and both small float64 gradient cases.
`make run RUN_ARGS=--small` bounds only the forward validation shapes.
All forward, component, and backward arithmetic executes in HIP IRIS tasks;
CPU NumPy long-double arithmetic is only an independent whole-output oracle. Every output and each
tiny gradient element is checked (the latter by central finite differences).

The original framework's GPU-generated tiny gradcheck inputs cannot be
reproduced without a working HIP PyTorch; these cases explicitly use the same
seed with CPU PyTorch instead. Forward CPU-generated tensors are unchanged.
The original native HIP sibling requires ROCm PyTorch, unavailable locally
(installed PyTorch is a CUDA build). No native HIP timing or parity is claimed.
This port replaces framework dispatch/accessors/autograd registration with
contiguous DMem buffers and explicit forward/backward tasks. Kernel equations
and reduction order come directly from the original CUDA implementation.
Framework dispatch overhead/performance is therefore not comparable.

Dependencies: IRIS with HIP backend, hipcc, Python, torch (CPU input creation),
numpy. No CPU fallback exists for benchmark computation.
