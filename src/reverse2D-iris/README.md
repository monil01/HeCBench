# IRIS 2D reversal

DMem HIP kernels cover all four unsigned element widths, row/column-major
layouts and row/column reversals. Original CLI and 8192x8192,100-repeat Makefile
defaults remain unchanged. Every output is checked against an independent
scatter-style CPU reversal. `HECBENCH_DUMP=path` appends each complete result in
native element representation (all four cases, then each type).

The native HIP input assignment was commented out, unlike the CUDA original.
Restoring the CUDA seeded generator (123, uniform 0..255) gives both models
the intended defined inputs. No kernel changes were needed for that repair.
The original native vector kernel also reversed each pair twice, undoing the
internal vector reversal. Its coalesced half-width indexing used scalar pairs
rather than vector pairs and skipped rows for odd vector counts. CUDA/HIP now
swap each pair once and round the half-width by complete vectors. These are
correctness repairs, not different reversal semantics. IRIS retains the
simple elementwise mapping, a performance portability compromise relative to
the native vectorized implementation.
Native CUDA/HIP now include the same whole-output checker and optional dumps.
Validation covers 1x1,3x5,16x16,24x48,257x513,1024x2048, all
16 combinations per size, with complete native/IRIS artifacts byte-identical.
The original 8192x8192 timing default was not run in this bounded validation.
