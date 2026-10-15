# slit: IRIS DMem + hipFFT

Preserves the double slit, repeated fresh-input in-place double-complex 2D
forward FFT, intensity calculation, and full radix-2 CPU reference check.
CLI remains `./main N repeat`; `make run` defaults to `1024 1000`.
The reference checker supports power-of-two N; invalid arguments are rejected.

The shared `tools/iris_vendor_bridge` dispatches actual hipFFT from an IRIS
task with DMem pointers and the IRIS-owned stream. Both the complex array and
exposed hipFFT workspace are DMem-owned. Plan auto-allocation is disabled;
opaque internal vendor plan state is retained. No native host GPU array
allocation/copy/launch or CPU FFT fallback is used. Each repetition refreshes
and uploads the original input, not the preceding FFT result. The final array
is flushed once. CPU reference computation is validation only.

Plan creation/destruction runs on the same selected IRIS worker as execution.
After destroying the plan, explicit `rocfft_cleanup()` releases vendor-library
resources before IRIS resets its HIP contexts. This avoids an observed
intermittent HSA heap abort at teardown; no runtime edit or skipped finalization
is used. Smaller validation cases exposed zero workspace bytes; the original
1024x1024 workload requested and used 17,824,768 DMem-owned workspace bytes.

Validation: seven native/IRIS pairs including the original `1024 1000` default,
3,366,912 output doubles byte-exact with native HIP and matching an independent
NumPy complex FFT/intensity oracle. Repeated small-process tests also exercise
vendor cleanup and IRIS finalization. AMD HIP validation only; CUDA is untested.

Build: `make IRIS_ROOT=/path/to/iris`; run with `IRIS_ARCHS=HIP` in this directory.
The callback times FFT plus stream synchronization, excluding IRIS upload,
scheduler and plan setup. Native times FFT plus device synchronization. Compare
whole-process timings separately; synchronization scope differs and no timing
or overlap equivalence is claimed.

Optional `HECBENCH_SLIT_DUMP=/path/file` writes all complex FFT values followed
by all intensities (native doubles). Identical opt-in CUDA/HIP hooks expose the
same complete output without changing inputs or default workload. Untouched
native and prior candidate snapshots remain in the local porting state.
