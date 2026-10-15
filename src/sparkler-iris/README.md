# Sparkler: MPI, IRIS HIP initialization, and genuine hipBLAS SGEMM

Retains the complete original FLOAT32 algorithm: rank partitioning, eight-wide
matrix padding, 499-prime sparse initialization, active-step rules, all MPI
barriers, N/T SGEMM, spot checks, original upper-pair checksum, MPI_Allreduce,
and timing output. The original default run is four ranks with
`--num_vector 4000 --num_field 90000 --num_iterations 10`.

Build with `make`; IRIS_ROOT, IRIS_BUILD, HIPCC, HIP_ARCH, MPICXX, MPIEXEC,
RANKS, LAUNCHER, and RUN_ARGS are configurable. `make run` uses those defaults.
For a smaller distributed test from this directory:

```
IRIS_ARCHS=HIP mpirun -n 4 ./main --num_vector 61 --num_field 1009 --num_iterations 2
```

All GPU matrices are IRIS DMem. The original initializer is an exported HIP
kernel submitted through IRIS. SGEMM runs via the genuine vendor bridge with
DMem-resolved pointers and the IRIS stream; outputs return through DMem flush.
There is no native allocation/copy/launch path or CPU GEMM fallback. All ranks
use device zero, matching the native source's unset/default device policy.
MPI_THREAD_FUNNELED initialization accommodates IRIS threads; MPI calls remain
on the main thread. No GPU-aware MPI or GPU collectives are required by source.

P1/performance: ordinary zeroed host storage replaces pinned host storage;
task submissions are synchronous, and each vendor callback creates/destroys its
own hipBLAS handle. These change transfer/dispatch/overlap costs, not arithmetic.
No shared runtime changes are required. The bridge and library search path can
be overridden with IRIS_KERNEL_HOST2HIP and IRIS_KERNEL_BIN_HIP when running
outside this directory.

## Whole-output validation

Set SPARKLER_DUMP to an output prefix to write every active result matrix in
every iteration on every rank. `observe.hpp` also checks each meaningful cell
against the exact LCM-count identity. Padding is excluded. This optional
diagnostic is identically inserted into an isolated native source copy; native
algorithm, GPU initialization, BLAS, streams, and MPI remain unchanged. The
original CUDA/HIP files are untouched. `verify.py` additionally compares all
rank files bit-for-bit, independently computes every cell with Python integers,
checks the exact active-block coverage, and reconstructs the reduced hash with
its original modulus/64-bit wrap semantics. No timing-only pass is accepted.

Validated four-rank uneven division (61 vectors, 1009 fields, two iterations)
and three-rank odd-process partition/padded matrix case (509 vectors, 1021
fields, one iteration), including wrap across the initializer's 499-stride
period. The production default 4000/90000/10 is retained but not certified by
these bounded runs. Invalid CLI inputs and dimensions overflowing BLAS int
parameters retain upstream limitations.
