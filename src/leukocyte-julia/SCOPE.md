# Leukocyte: source-defined complete detector and tracker

The named configuration runs the authentic original 640×480, 600-frame AVI. Each trial executes complete detection plus tracking for 1, 3, and the original default 100 frames. Detection uses seven 150-point circles, dilation a radius-12 circular mask, CPU spline refinement, and all detected cells. Tracking uses the original CPU subimage extraction/normalization, three-stage GPU pipeline, adaptive MGVF iteration (maximum 500, cutoff 1e-5), and CPU ellipse evolution. No clinical detection accuracy, annotated ground truth, or biomedical model certification is claimed.

## Conditional reference

The immutable original source is preserved. Four explicitly scoped corrections define the secondary reference:

1. Initialize the device GICOV border to the already zeroed host-result policy. Original device allocation left it undefined.
2. Add a CTA barrier after virtual-tile neighbor reads and before shared writes, preserving intended old-state stencil dependence. The inherited HIP/MI100 witness is historical, not a current original-CUDA failure.
3. Reject invalid bilinear coordinates, with no clipping or interpolation formula change. A current original CPU ASan invalid-input control reproduces the missing bounds protection; this does not establish a crash on the complete valid AVI run.
4. Use integer division for row-major row indices. A fresh CUDA/V100 coordinate probe found 44 wrong row positions at valid widths 55/61; an independently checked CUDA old-versus-corrected one-step operator control found 19 invalid old outputs and zero corrected failures under its predeclared criterion.

Original host policies are retained, including approximate PI=3.14159, Float32 angle construction, original spline control orientation/update, and source asymmetric Y2 formula. These are source policies, not silently repaired mathematical models. This conditional reference is excluded from untouched-native aggregates.

## Measurement and evidence

Warm all three full cases before five captured trials. Retain the mean of trials 2–5. The measured quantity is the sum of 110 original kernel/completion regions: three GICOV, three dilation, and 104 MGVF launches. A uniform iteration-counter store per cell is included. CPU processing, video decoding, memory setup, guard/input/output capture, and all checking are outside each kernel ROI. Instrumented driver wall time is separate and is not the unmodified allocation-inclusive CLI time.

Per trial retain the complete external packed-video/constants/initial-gradient prefix (6,329,529 exact values), all GPU stage outputs and endogenous flow inputs, CPU detected centers and every frame's 20-radius cell trajectories. Actual pre/post physical device buffers, 256-byte prefix/suffix guards, decoded video bytes, 32-byte host guards, kernel regions, and per-cell iteration counts are retained separately. Iteration counts are work covariates, not forcibly substituted outputs.

Independent equations check every initial GICOV pixel with FP64 two-pass variance, exact CPU dilation, and the first actual patch of every distinct shape with a FP64 discrete MGVF replay at its observed iteration count. This checks selected fixed-count fields, not independent convergence or every complete trajectory. Matrix left-inverse, affine interpolation, constant-field and varied-width one-step controls provide additional witnesses. Bounds and tolerances were declared before the full pilot. Source policies and conditional provenance remain visible even when numerical agreement passes.

## Translation state

CUDA.jl translates both the CPU pipeline and authored GPU kernels. Only the original AVI decoder is reused through a narrow CPU C ABI. CPU libm primitives are explicitly bound to libm.so.6 to preserve native input construction; a failed unqualified-sinf control is archived. All source/program/feedback attempts and rejected complete numerical series remain recorded. No framework expressibility conclusion follows from these failures.
