# Rodrigues verification scope

The unchanged native rotate and rotate2 kernels operate on packed 12/16-byte records. Their sign convention is passive rotation; the fourth float4 word becomes positive zero. The historical Julia marker stub is preserved in reference_history.

Twenty-four cases use 4099 signed/unit/zero/basis points, the source-rounded oblique axis and exact z axis, angles 0, 0.5 and Float32(pi/2), with one or seventeen in-place steps. This bounded fixture does not certify arbitrary axes, all angles, the source host random generator, or large-scale throughput.

A separately generated Float64 quaternion rotates by negative R*actual Float32 angle. A predeclared absolute budget 5e-6+8e-7*R accommodates Float32 axis normalization, SFU and repeated arithmetic. Full xyz, norm and axis-projection controls, exact actual input prefixes, guards and positive-zero fourth fields are checked in every trial. Device parameter echoes are guarded; native consumes host by-value parameters while Julia reads its buffer, a disclosed ABI covariate.

Timing sums 24 individual host launch/context-completion windows, each divided by that case's R. Reset/allocation/copy, JIT, oracle and capture are outside. Retain five checked trials and average only trials2–5, with each path retained separately. No measured traffic, bandwidth, vectorized-codegen or roofline claim follows from packed layout alone.
