IRIS DMem LIBOR port. Original constant path inputs, NPATH=96000, NN=80,
NMAT=40, block=64/grid=1500 and default repeat 100 are preserved. Both kernels
retain the CUDA formulas and fast intrinsics as in native HIP. Portfolio values
and sensitivities are flushed after their respective repeated kernels; inputs
remain iris_r and full output buffers iris_w. Original 224.323 portfolio and
21.348 sensitivity checks use 1e-3 tolerance; nonfinite means fail explicitly.
Bounded validation repeats 1/2, with the original full path count.
Oxford copyright/BSD3 notice retained in host and kernel sources.
