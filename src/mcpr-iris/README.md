IRIS DMem Monte Carlo probability port. The path/repeat arguments, default
alphas.csv 10, n=26280/K=21/M=10000 and mt19937 seed 19937 are preserved.
make run extracts the provided compressed dataset locally. Dataset output is
not source and must not be committed. Each kernel flushes probs for its original
1e-3 reference check. Transposed host arrays refresh DMem with update(host).
The third kernel uses static shared storage sized identically to the original
fixed K=21 and block size 96. All three kernels execute on an enumerated HIP GPU.
Repeat 1 is the bounded case; dataset dimensions are fixed in the baseline.
Nonfinite results fail, and any reference failure produces nonzero exit status.
