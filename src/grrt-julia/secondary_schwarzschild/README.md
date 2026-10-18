# Conditional Schwarzschild follow-up

This is a separately named numerical-method configuration (`configuration_role: corrected_reference_secondary`). The original GRRT configuration, sources, all failed independent checks and GPU artifacts remain unchanged and blocked.

## What changes

The source null-reduced radial equation is replaced with its canonical Hamiltonian extension for spin A=0. They agree on the intended exact null manifold, but differ off that manifold. The change is justified by exact rational algebra and an analytic radial-null witness, not by declaring the astrophysical model invalid. Disk crossing is localized with48 bisections. Transfer uses stable `expm1` and adaptive fourth-order exponential maps, with positivity/finite acceptance and explicit failure-to-NaN propagation on GPU. This changes the numerical method and its cost. It is not an original-source speed comparison or a general Kerr port.

## Accuracy contract

Two complete8×8 operators use the same source observer, disk/horizon/escape boundaries and declared constants. The CPU oracle uses true long-double math overloads (mantissa64), controller scale1e15 and the conditional method; its128 observables are rounded to Float64 only for the binary comparison. Full CPU convergence includes Float64 scales1e12/1e14/1e15 and long-double scales1e14/1e15, all within unchanged atol1e-8+rtol1e-6. The ld15 table row is the reference, not an independent experiment. Complete coordinate/observable triples and exact eight-value input prefixes are retained.

## Measurement

All models execute both complete adaptive image kernels with100×1 threads and1×8 blocks. Geometry, event localization, transfer and their control loops are timed together; allocations, oracle construction, input transfers and artifacts are outside. Compile warmup precedes five in-process trials; the mean uses trials2–5. Sparse64-ray occupancy and the changed solver prevent original-CLI or throughput conclusions. No hardware bottleneck is inferred without counters.

The native adapter's inherited `initial_null_residual_CPU:0` raw telemetry field is a placeholder after replacing runtime CPU-gold generation with a saved extended-precision oracle. It is not measured and is excluded from scientific null-constraint claims. The actual radial witness and Julia CPU initialization check provide separate numerical evidence. Raw fields and samples remain preserved.

## Provenance and accounting

Scientific helper work reuses the original GRRT transcription and CPU reassessment. Costs are ordered marginal costs, not cold framework costs. Historical failed GPU trials and CPU prototype compile/timeout feedback remain separate from this conditional method's device candidates. See parent `secondary_method_reassessment.json`, `secondary_observable_convergence.json` and `secondary_radial_null_witness.jsonl`; raw CPU outputs remain in `.porting-state/portable_julia/`. No astrophysical physical-model validation is claimed.

## Explicit oracle precision boundary

State-dependent math uses explicit long-double std overloads (mantissa64), while original Float32 ISCO and literal-only constant precision remain. The scalar solid-angle `atan` still uses the global Double overload; its actual relative luminosity-prefactor difference from extended `std::atan` is7.52536e-17 for this fixture. This scalar witness is not a global trajectory error bound. Saved gold and execution remain immutable. The inherited CPU-origin comment in the generated CUDA header is historical; actual `__device__`/`__global__` definitions implement the controlled secondary method described here.
