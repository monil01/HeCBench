Original native HIP crashed inside unchecked bilinear sampling during initial
spline detection. A fresh diagnostic with explicit shape/finite/domain checks
completed36-cell detection and one full tracking frame, without triggering any
invalid-coordinate check. The old crash's deeper cause is not claimed proven.

CUDA/HIP find_ellipse.c now checks shape equality and finite coordinates whose
four interpolation neighbors exist. Invalid input fails, not clips. For valid
input, the original bilinear formula, spline equations,40-pixel convention,
thresholds, iteration counts and all defaults are unchanged. Independent affine
plane sampling validates the formula, and an invalid coordinate must fail.
Untouched original sources and the earlier crash history remain local.

Optional HECBENCH_LEUKO_VALIDATE/HECBENCH_LEUKOCYTE_DUMP hooks record all GPU
stage fields and every frame's centers/radii. Native validation only adds dumps;
it retains original HIP kernels. The IRIS port releases detection DMem bindings
before freed host storage can be reused by tracking; all API failures are checked.

Attempts05/06 identify a real out-of-domain sample y218.243 in219row gradient.
An experimental spline-control update correction did not change that failure
and is NOT shipped; original spline equations/update policy remain unchanged.
Its complete diagnostic source/zero-force test remain preserved in attempt06.

Native d_gicov was allocated but never zeroed/uploaded, while its kernel writes
only the interior22pixels away from each border. The zeroed host_gicov array
did not initialize device borders, allowing stale GPU data to create spurious
edge detections. Initialize the complete device output to the same intended0
as the host and IRIS DMem. Full-field verification independently requires all
noncomputed border responses to be exactly0. No image/input/threshold changes.

MGVF also needs a barrier after every warp has computed its current tile, before
any warp writes the previous shared-memory tile. The old end-of-tile barrier
does not prevent write/read overlap among warps during computation. Native and
IRIS had identical canonical LLVM kernel bodies and identical inputs, but79
output floats differed up to3.87e-7. The additional barrier in all three sources
defines the intended Jacobi ordering; no floating-point tolerance is relaxed.
It costs one extra block barrier per virtual tile. A one-step unit loads a real
captured cell patch and compares both GPU paths to independent Jacobi equations.
