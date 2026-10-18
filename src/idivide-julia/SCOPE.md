# Exact signed fast division: observable named workload

Historical CUDA.jl was a marker-only stub with fabricated timing text and
zero divisor counts; its original bytes are retained. New implementations
compute every signed quotient and remainder for4105Int32 inputs and28
positive/negative divisors, with both ordinary division and the genuine
multiply-high/magic-number algorithm. All20 full captures are bit-identical
and match independent Python arbitrary-precision abs// truncation and
n-q*d. Actual inputs, passed divisor fields, readonly data, and all guards
are exact. The unchanged native functional check also executes on+/-0..4095
for28divisors, outside timing; actual error words and guards are retained.

The timed region is56 observable quotient/remainder map launches per
repeat (R5), with source fastdiv.h unchanged. It adds explicit per-item
output stores in the native capture adapter and matched Julia maps. This
is not the original dummy0 throughput/latency timing suite, whose kernels
expose no numerical result. Those original timing kernels are not ported or
certified by these timings. Results are excluded from untouched-native
performance aggregates. No scientific algorithm repair is claimed.

An initial possibility of dead arithmetic in the original timing kernels
was investigated. Offline CUDA12.0 PTX retains div/multiply arithmetic;
SASS also contains arithmetic. This contrary static evidence is recorded;
we do not call the original timing kernels empty or infer dynamic instruction
counts from static instructions. Original functional and timing headers are
unchanged; no original huge exhaustive CLI coverage is claimed.

Scope excludes divisor zero, INT_MIN and INT_MIN/-1 overflow, and does not
prove all32-bit inputs/divisors. Negative quotient truncation is toward zero,
not floor. Fixed-width arithmetic and native compiler behavior are scoped
to tested inputs/toolchain. Magic setup, allocation/reset/readback/checker
and oracle are outside ROI. Five runs all checked; mean uses2..5. Small
arrays/56 dispatches are not an intrinsic instruction-throughput result or
measured roofline position. Atomix is used for the portable error counter,
which is diagnostic only, outside timed capture kernels.

## Matched setup revision

Native first adapter rebuilt magic descriptors inside ROI vs Julia precompute outside. Preserved original numerical-valid outputs and hash-bound primary timing exclusions. Native descriptor setup moved outside; freshall4 five-trial series selected by protocol, not speed. GPUkernel bodies/acceptance/inputs unchanged.
Both native and Julia now precompute all magic descriptors outside ROI.
Each Julia model retains one GPUcandidate, evaluated twice; shared measurement
work and device compiler-feedback repairs are separate (zero latter).
