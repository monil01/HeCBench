IRIS DMem quantization port preserving token/hidden/repeat arguments, seed123,
default4096/5137/1000, rounding-to-even/saturation, static and dynamic scaling,
zero-point variants, and FP16/BF16/FP32 reference byte comparisons.

Twelve HIP template instantiations have exported C symbols. Host allocations,
copies, and launches use DMem/task APIs; outputs flush before reference checks.
hipCUB reductions and HIP scalar types match the native sibling.

Both models pass all12 stage-specific byte comparisons at4tokens, hidden64
and513,2repeats. Per-stage observational verdicts prevent cancellation in the
original sum-of-memcmp aggregate from concealing a stage failure. The isolated
native validation copy prints the same observations without changing results.
