MDH IRIS dmem port
==================

Preserves three source Debye-Huckel vector kernels, default natom=5877/ngrid=134918, pre1/xkappa constants, random initialization and workgroup/iteration flags. Restores implicit srand(1) after IRIS init. All arrays are dmem; float/int scalar widths retained; output flushed for every variant. Kernel dynamic shared array is represented by a fixed 5*1024-float workgroup-local array, covering HIP/CUDA maximum block size without changing indexing or computation (P1).

Meaningful finite validation compiles HECBENCH_MDH_VALIDATION: natom=37, ngrid=1030, pre1=1.0. The original default overflows exp(kappa*natom) and the original absolute CPU test can mask paired infinities; therefore defaults alone are not numeric evidence. An isolated native snapshot adapter changes only these bounded test inputs and adds optional full float output capture, preserving the algorithm and original siblings/defaults. Tests all three native/IRIS kernels at -itmax 1 -wgsize 128; source CPU verdicts must match and all 1030 output floats/kernel must be finite and match at atol=rtol=1e-6. Optional HECBENCH_DUMP writes complete per-variant arrays.
