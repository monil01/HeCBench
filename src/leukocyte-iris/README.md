Leukocyte IRIS port
===================

The three original CUDA kernels (GICOV, dilation and iterative MGVF) run as
IRIS HIP tasks using DMem. Original host detection and snake tracking code
and Meschach routines are compiled from the read-only CUDA sibling into local
objects. SOURCE_DIR, MATRIX_DIR, IRIS_ROOT and IRIS_BUILD are configurable.
The local find_ellipse.h removes an unused CUDA runtime header dependency.

`make run` retains the original video argument and 100-frame default. Use
RUN_ARGS to select a recovered video path or a bounded frame count. Existing
OUTPUT coordinate reporting is enabled by default for artifact verification;
OUTPUT= disables it. No tracking computation is changed by that define.
