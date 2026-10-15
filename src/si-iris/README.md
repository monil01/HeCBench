# Set intersection through IRIS DMem and hipBLAS

Preserves the CUDA/HIP set one-hot encoding, CPU exclusive scan, upper-triangular
tile traversal, SGEMM, pair enumeration, CLI, binary format, and output text.
`--partition` defaults to 10000. `make run` retains the upstream generator
arguments (1000 sets, universe 1000, 100 draws) and ascending/descending runs.
The original per-set RNG reset to 19937 is intentionally unchanged.

Build with `make`; `fetch_deps.py` fetches the exact four original CMake pins
into untracked `.deps`. Boost headers are needed. IRIS_ROOT, IRIS_BUILD, HIPCC,
CXX, and CXXFLAGS are configurable. Usage from this directory:

```
IRIS_ARCHS=HIP ./main --input dataset.bin --output result.txt --partition 7
```

For another working directory set IRIS_KERNEL_HOST2HIP to the absolute path of
libhecbench_vendor.so. All device buffers belong to IRIS DMem. Each task invokes
genuine hipBLAS SGEMM on IRIS-resolved pointers and stream, then flushes output.
No native allocation/copy or CPU GEMM fallback is used. P1: callback handle
creation, synchronization, DMem transfers, and dispatch add overhead relative
to the original persistent-handle implementation.

## Necessary baseline bug repairs

Original CUDA/HIP sources remain untouched; validation compiles an isolated
native copy with these same minimal repairs:

* Transposed one-hot storage and the result writer use `partition` row stride.
  Original SGEMM incorrectly passes `B.length` for both lda and ldc. Set both
  to `partition`; m/n still use actual tile lengths. Full tiles are unchanged.
* Uniform/Zipf generators draw labels 1..universe but address a zero-based
  bitset of length universe. Subtract one before those two position writes;
  serialization already adds one. Draws/seeds/distributions are unchanged.
  Other distributions are unchanged. Normal's original unbounded draws remain
  an upstream input-domain limitation.

## Validation

Native HIP and IRIS execute real hipBLAS SGEMM on MI100. `verify.py` checks
every output pair against native and independent Python set intersections,
rejecting missing/duplicate pairs. Validated original 1000-set ascending and
descending uniform cases and a 31-set heterogeneous dataset concatenating
unchanged source-generated Bernoulli sets at probabilities .1/.3/.5/.7 with
partition 7 (short tail tile). Original default sets are intentionally identical
because the generator resets its seed per set; the heterogeneous case protects
against validating only that degenerate pattern. No external input dataset or
fabricated native output is used. Invalid input files, arbitrary huge partitions,
and all possible normal/Zipf parameter domains are not certified.
