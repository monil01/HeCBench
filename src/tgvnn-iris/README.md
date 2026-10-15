# TGV + nuclear-norm dynamic MRI reconstruction through IRIS DMem

Full port of the CUDA primal-dual reconstruction, not an elementwise subset.
Retains the original 27 kernels, gradient/adjoint boundary formulas, alpha
schedule, FFT sampling/adjoint, complex SVD, singular-value threshold, complex
diagonal multiplication/GEMM, both L/S outputs, and magnitude-based SER checks.
The original MIT copyright/license notice is retained in main.cpp; RA format
provenance remains in ra.cpp.

## Build and run

```
make
IRIS_ARCHS=HIP make run
IRIS_ARCHS=HIP ./main -i 3 -o recon.ra ../tgvnn-cuda/data/perfusion.ra ../tgvnn-cuda/data/perfusion_mask.ra
```

Run from this directory, or set IRIS_KERNEL_HOST2HIP and IRIS_KERNEL_BIN_HIP to
absolute paths for libhecbench_vendor.so and kernel.hip. IRIS_ROOT, IRIS_BUILD,
HIPCC, HIP_ARCH, CXXFLAGS, and RUN_ARGS are configurable. Requires hipFFT,
hipSOLVER, hipBLAS, and the committed tools/iris_vendor_bridge implementation.

Default data are the original pincat image/mask; default output pincat_recon.ra;
500 iterations; alpha .004, beta .5, sigma/tau .25, mu 1; reduction .5;
grid/block 128/128. Original options -i/-a/-b/-s/-t/-m/-o/-G/-B and long
equivalents are retained. The upstream -m switch falls through to -o; when
setting mu, supply -o afterwards (as in validation). This quirk is not silently
changed. The original --help exit code 1 is also retained.

## IRIS memory and vendor interoperability

All caller device arrays, scalar device results, SVD work/rwork/devInfo, and
FFT work area are IRIS DMem. Host pointer tokens preserve the source's offset
expressions but are never native device pointers. Kernel tasks pass DMem bases
and explicit byte offsets, not DMemRegion. Device copies/zeroing are IRIS HIP
kernels; host transfers use DMem update/flush. No native allocation/copy/launch
path or CPU reconstruction fallback exists in the benchmark.

Vendor callbacks run true hipfftExecC2C, hipsolverDnCgesvd ('S','S'),
hipblasCdgmm_64/Cgemm_64, Icamax_64, and Snrm2_64 on DMem-resolved GPU pointers
and the IRIS stream. FFT auto-allocation is disabled; its queried work area is
DMem. Solver and BLAS handles are reused, rebound to the managed stream, and
destroyed through callbacks. Every SVD devInfo is flushed and checked. Callback
errors fail loudly through the shared bridge. The installed HIP 7.0.1 supports
the original _64 operations; no 32-bit vendor API substitution was needed.

P1/performance differences: synchronous IRIS tasks impose explicit ordering
instead of the source's mixture of default/auxiliary streams. Host shadows and
offset bookkeeping add memory/dispatch overhead; D2D copies use byte-copy
kernels rather than native asynchronous DMA. Every SVD convergence result is
checked. Extra stage reads occur only with TGVNN_DUMP. These costs are included
in measured validation timings; they are not isolated vendor kernel timings.

The adapter range-checks dimensions/counts against INT_MAX (the original
FFT/SVD interfaces already use int) and accepts rows/cols/ndyn >=3 with
rows*cols >= ndyn. All three original datasets fit this supported domain.
Malformed RA files, oversized/unrepresentable dimensions, and mathematically
invalid stepsizes are not certified.

## Necessary native HIP baseline repair

All original CUDA and HIP sources remain untouched. Isolated native validation
copies change exactly the arraydot product to `cmul(alpha*a,b)` and add
diagnostic stage reads plus a stream synchronization before writing output.
CUDA float2math.h defines a complex operator*(float2,float2). The HIP sibling
renamed it cmul but left arraydot using *, which selects HIP's componentwise
vector multiplication and erases imaginary Fourier samples under a real mask.
The explicit cmul repair restores the original CUDA formula. The unchanged HIP
executable was also built/run as evidence, but its known-wrong reconstruction
is not used as the correctness target. On pincat at one iteration its original
SER is 9.73 dB; the CUDA-faithful repaired native and IRIS results are 21.59 dB.
The corrected result is independently checked, not justified by SER alone.

## Whole-output validation

`verify.py` is a CPU-only validation tool, never a benchmark execution path.
It checks every complex value in both RA output components against native HIP
(atol 2e-5, rtol 2e-4, relative L2 <1e-5), reconstructs the FFT sampling/adjoint
with NumPy, independently validates the full complex SVD/shrink/BLAS proximal
map, reconstructs both SER values, and checks actual successful callback counts.
All GPU values must be finite. Independent source-oracle tolerance is atol
3e-5, rtol 2e-3 and relative L2 <3e-4, including near-zero outputs.

Full original-size cases: pincat 128x128x50 (one iteration), perfusion
190x90x70 (three iterations; alpha .006, beta .35, tau/sigma .2, mu .8,
grid/block 37/64), and breast 192x192x105 (two iterations; grid/block 71/256).
These compare the complete multi-iteration NumPy primal-dual reference as well
as native GPU outputs; every stage and output passes. The temporal adjoint's
original unscaled endpoint formulas are intentionally retained in the oracle.

The original default pincat 500-iteration workload also passed: all 1,638,400
complex L/S values bit-exact against repaired native HIP, SER 32.74 dB, and
actual 501/502 forward/backward FFTs plus 500 each of SVD/Cdgmm_64/Cgemm_64.
This long case checks complete GPU outputs and independent initial FFT/proximal
stages, but does not run a full 500-step NumPy loop. Across all four cases,
13,412,240 complex L/S output values are checked (all native/IRIS bit-exact).
On this MI100/HIP 7.0.1 setup native/IRIS default wall times including diagnostics
were 12.36/57.97 seconds; observed overhead is consistent with the synchronous
task/copy adaptations above, not a claim about isolated library performance.

TGVNN_DUMP=PREFIX writes normalized, sampled, zero-fill, first proximal input,
first proximal output, and shrunk-singular-value binaries from actual GPU
buffers. The identical optional observer is inserted in the isolated native
copy. Given those diagnostics and native.ra/iris.ra in DIR:

```
OPENBLAS_NUM_THREADS=1 python3 verify.py DIR IMAGE.ra MASK.ra --iterations 3 --alpha .006 --beta .35 --tau .2 --sigma .2 --mu .8
```

For long runs --skip-full-cpu skips only NumPy's complete iteration loop;
whole native/IRIS output, independent first FFT/proximal stages, both SERs, and
exact vendor-call coverage remain mandatory. GPU execution is bounded and
serialized per native/IRIS pair; all CPU verification runs outside that lock.
