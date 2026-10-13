# MM-CSF IRIS dmem port

Retains original sparse tensor partitioning, all seven 3-D/4-D device kernels,
mode transitions, scalar widths (ITYPE=size_t), CLI/defaults, CPU COO reference,
and matrix initialization. Device arrays are iris::DMem objects; original pointer
offsets become explicit size_t byte offsets applied in GPU exported wrappers.
The installed HIP runtime shares one archs_off slot per memory object, so native
offset API parameters would alias different views of dU. Wrappers call the
unchanged original kernel bodies with correctly offset device pointers.
Partial writes use iris_rw.
Host resets/uploads flush prior device state and call DMem.update.

Build with make, test with make test, run original generated workload with make run.
IRIS_ROOT defaults to /tmp/iris. Requires HIP and Boost header-only CPU sort/hash.
HIP device selection reports backend3/type256 AMD and rejects CPU fallback.
kernel.cu retains CUDA-compatible device arithmetic (not validated locally).

HECBENCH_OUTPUT optionally captures the entire final-mode GPU matrix before
the unchanged original CPU check; no algorithm/default changes. Isolated native
validation uses the same capture-only adapter. Validation uses original toy.tns
and bounded deterministic CLI 3-D/4-D tensors. Original default tensor generator
arguments are unchanged.
