Preserves CUDA command line, defaults, seed 19937, int8/int4 packing,
half precision scalars, vectorized kernels and launch geometry. HIP kernels
reuse the native sibling with C linkage. Host memory uses dmem exclusively.
The optional FP16 test remains disabled as in CUDA main. CPU validation checks
every row using the original 0.125*n/512 tolerance; HECBENCH_DUMP=1 emits
every result for comparison. Isolated native validation adds only result dumps;
native siblings and default algorithm are unchanged.
HIP C-linkage kernels specialize the native template for WarpSize=64 on MI100.
The native default at size 512 reports a tolerance diagnostic for int8 row 0
(619.5 versus 619.653564). Its nonfatal diagnostic behavior is preserved.
Bounded passing validation uses the existing -u option with scale 0.0078125;
default scale 0.0625 is unchanged.
