# Leukocyte completed

Original authentic video, 36 cells, 1/3/full-default100 tracking frames. All detector and iterative MGVF input/output fields byte-exact and every frame center/20radii exact: 26167980 finite values. Independent affine-plane interpolation and 3321-value first-cell one-step Jacobi oracle pass, maxabs5.063e-8. Native CUDA/HIP minimal repairs zero unwritten GICOV borders, guard invalid bilinear coordinates (no clipping), and synchronize shared-memory Jacobi reads before tile writes; equivalent IRIS barrier. Additional barrier has performance cost. Complete algorithm/defaults preserved; native and IRIS actual GPU launches only.

Files kernel.hip.cpp, iris_helpers.h, README.md, track_ellipse_gpu.cpp, Makefile, find_ellipse.h, verification.h, verify_mgvf_step.py, main.cpp, interpolation_reference.cpp, BASELINE_FIX.md, mgvf_reference_driver.cpp; native detect_main.c, find_ellipse.c, kernel_IMGVF.h, track_ellipse.c in CUDA/HIP.

Timings {'hip_compile_seconds': 5.99, 'hip_runtime_seconds': 3.641454908531159, 'iris_compile_seconds': 11.92, 'iris_runtime_seconds': 4.764410950709134}

Build make -B -j4; validate_expanded.py per native/IRIS pair flock-w600/timeout180, CPU checks outsidelease. validate_step.py flock/timeout60. Attempts02 diagnostic;03/04 Meschach link fixes;05/06 invalid interpolation;07 native-exact inputs but shared race mismatch;08 synchronized pass. All history and untouched baselines preserved.
