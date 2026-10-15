# Rayleigh–Bénard convection with IRIS DMem/HIP

The complete source finite-difference, spectral transform, adaptive-dt and
three-stage Runge–Kutta sequence is preserved, with the original M=500,
N=1000, Rayleigh number, initialization, timestep CLI and make-run1000steps.
All seven original GPU kernel bodies are retained; their signatures gain
base-buffer offsets resolved by IRIS tasks. Every Scopy/Sscal/Saxpy/Sgemm/
Sdot/Isamax is still the real hipBLAS operation on an IRIS-owned stream via
the shared vendor bridge. No vendor arithmetic was reimplemented on the CPU.

dmem_ops.hpp keeps original pointer arithmetic as offsets into host-shadow
buffers owned by iris::DMem. Those host addresses are NOT device pointers;
callbacks receive actual GPU pointers exclusively from IRIS task arguments.
The adapters never call native GPU allocation/copy or native kernel launch.
They use whole DMem roots plus scalar offsets, not DMemRegion, so they do not
depend on the live runtime region-ownership patch. Vector initialization uses
DMem.update; every output read uses a flush_out task. Dot/argmax scalars are
computed into GPU DMem in device pointer mode and flushed to the original
host scalar, rather than masquerading as host scalar computation.

P1: synchronous per-operation tasks, full-root coherence and bridge calls
add overhead; native stream throughput or original kernel-only performance
is not claimed. The original numerical sequence, defaults and physical model
are not changed, and the native source remains untouched.

RBC_DUMP=/absolute/path.bin enables diagnostics in both port and isolated
native reference copy: save the last step's input and dt, then dump every
element of ten state/work fields plus omega, adaptive dt and a final call of
the original complete Nusselt calculation. The latter diagnostic exercises
boundary BLAS, elementwise kernels and dot reductions even in short tests.
verify.py compares all5975007 finite floats (rtol1e-5/atol1e-4), and separately
checks every final temperature value against an independent FP32-rounded
Runge–Kutta update of the dumped prior field and GPU stage derivatives.
Diagnostics do not replace any benchmark GPU output. Production-default
1000steps are retained; validation uses1 and3 steps at the FULL original grid.
All11950014 dumped floats matched native HIP bit-for-bit across those two runs;
the3-step last update changed497965 temperatures and its independent RK3
identity was exact. CUDA/HIP source comparison confirms executable numerical
operations differ only in backend API spelling; untouched-source diff is saved
with the local evidence.
