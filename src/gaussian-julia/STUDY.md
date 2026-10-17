# Gaussian elimination study

## Genuine reference and scientific scope

The historical Julia program only incremented a bounded zero array and printed unconditional PASS. It is preserved as `gaussian-julia-cuda.jl/original_main.jl`. A fresh genuine reference implements the original two-kernel nonpivoted Gaussian elimination: Fan1 computes column multipliers; Fan2 updates trailing matrix/RHS, with only the column-zero lane updating each RHS. Host backward substitution remains outside the GPU timed region. This is a dense linear-system primitive; the source establishes no particular PDE or scientific dataset.

All4 models use a shared checked-in row-major128x128 Float32 exponential Toeplitz fixture and RHS1, multiplier matrix zero. The canonical fixture is generated as Python Float64 10*exp(-.01*distance), then roundedFloat32. It follows the original source matrix family; no byte-identical native-libm generator claim is made. The source4096 size, file-input matrices, singular and arbitrary ill-conditioned cases remain unmeasured. Nonpivoted stability outside the named fixture is not certified.

## Numerical evidence

Every trial checks all33024 scalar outputs: full16384-element A,128 RHS,16384 multiplier matrix, and128 host solution. Each model checks full intermediates against the original CPU recurrence with1e-4 absolute tolerance and solution with original1e-3. Both references' complete artifacts are also compared. All outputs must be finite. Julia additionally compares against independent pivoted Float64 Ax=b and all models check normalized infinity residual below1e-5. Measured native/Julia maximum fullstate difference is6.21676445e-5; Julia solution error against Float64 is4.54e-6, scaled residual8.55e-9. No ISA-level attribution of rounding differences is made.

## Mapping and timing

All4 retain source row-major buffers, Fan1 one256-thread block and Fan2 grid8x8 of16x16 threads, column=x/row=y, shrinking active bounds eachpivot. There are254 ordered kernels per complete128x128 forward pass. The same GPU-forward phase is timed once pertrial, with explicit JIT warmup and5 synchronized in-process trials; mean runs2–5. Reset/H2D, allocation,D2H,Float32 BackSub,Float64 solve/residual and artifact writes are outside. Actual device state128.5KiB fits V100 L2, so this does not establish DRAM/application scaling. No bottleneck is inferred without profiling.

KA uses linear/NTuple globalindices with no shared scratch. JACC uses separate explicit 1D/2D LaunchSpecs and shmem_size=0, applying the prior FWT mutable-grid lesson before first draft. KA and JACC each passed candidate1/check1 with no repairs. Current genuine CUDA.jl reference is one separately costed semantic repair and passed candidate1/check1. Historical generation cost is unknown. Prospective consumed-usage windows include cache/context and coordination; broad workflow overlaps child windows and is not additive or monetary billing. Partial saved-source proxies remain separate.
