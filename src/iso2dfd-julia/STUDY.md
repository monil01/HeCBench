# ISO2DFD wave-propagation study

The original CUDA and CUDA.jl wave kernels remain unchanged. KA/JACC express the same rectangular two-dimensional second-order acoustic stencil and alternating time-level recurrence, with16x16 workgroups. The fixture uses the original12 wavelet constants/overwrite loop, constant velocity-squared2250000, DT0.002 and DXY20, on128x160 cells for24 steps. Initial Float32 constants and operation order agree across fixtures.

The standalone native and inherited Julia checkers always read a particular buffer even when parity makes it the preceding rather than final state. This harness checks both actual final24 and preceding23 states, including untouchedzero halos; it does not patch the native device kernel. All40960finite outputs require atol1e-4+rtol1e-5, stronger than the original interior-only0.1 check. Agreement does not establish physically absorbing boundary conditions or convergence at arbitrary timestep/mesh scales.

Warmup/JIT precedes trial1. Reset copies lie outside timing. Every trial advances24 ordered steps, synchronized at boundaries, with time reported per step. Five correctness checks and all samples are retained; mean excludes trial1. CPU and both GPU references confirm arrays. No profiles or bandwidth mechanism have been measured. Historical translation tokens/attempts remain unknown; new permodel source-only token proxies are partial.

## Measurement setup deviation

The initial three-model KA bundle remains archived. An unsupported per-model reuse_run key was ignored, so the second invocation executed allfour models. The first completefour-model bundle is primary bycoverage, not its timings; allpreliminarysamples remain retained. T1 measurementconfiguration error, no kernel/accuracyrepair, no further GPUrerun.
