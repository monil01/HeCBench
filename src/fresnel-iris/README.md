IRIS DMem port preserving the original Fresnel sine-integral implementation,
including its original formulas and Chebyshev/asymptotic branch arithmetic.
This port verifies equivalence to native HIP; it does not repair native math.

`make run` retains100 repetitions,80000000 points and interval1e-7. Optional
HECBENCH_FRESNEL_POINTS bounds validation to evenly spaced points over[0,8].
The131073-point validation includes exact0.5,1,3,5,7 boundaries, their neighbors,
all original approximation branches and an incomplete final256-thread block.
The same opt-in input bound is applied to an isolated native host fixture;
native kernels and repository siblings are untouched.

HECBENCH_FRESNEL_DUMP writes doubles for comparison. With
HECBENCH_FRESNEL_CAPTURE_ONLY set, the program captures actual GPU output and
explicitly defers verification, printing no PASS. Validation compares every
native/IRIS result and the CUDA-derived CPU reference outside the GPU lock at
the original absolute tolerance1e-6, rejecting nonfinite values. No CPU fallback
is used for computation. The original80-million-point density is not certified
by this bounded check; full-range piecewise arithmetic is covered.
