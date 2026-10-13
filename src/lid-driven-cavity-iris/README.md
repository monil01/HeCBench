Retains all eleven CUDA finite-difference kernels, double precision constants,
pressure convergence test, boundary conditions, final velocity artifact and
defaults NUM=512/BLOCK_SIZE=128/time_end=0.001. Ten buffers use dmem and rw
preserves partial writes. Reduction buffers flush before host convergence/CFL
calculations. Final velocity and pressure arrays flush before artifact output.
VALIDATION_NUM=64 VALIDATION_BLOCK=16 bounds deterministic local checks;
the isolated native copy applies these compile options and a complete binary
dump only. All velocity/pressure values and velocity_gpu.dat are compared.
