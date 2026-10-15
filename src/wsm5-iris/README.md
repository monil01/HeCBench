IRIS dmem port retaining the full original WSM5 microphysics kernel, constants,
60x46x3 dimensions,8x8 launch, MKX4, timestep10 and uniform0.001 initialization
for every repetition. All18 integer bounds and float timestep use exact scalar
widths. Output arrays use rw and are flushed in full; repetitions use fresh
dmem objects and preserve the source reset semantics.

WSM5_DUMP optionally saves all11 modified arrays after every repetition.
Isolated native validation adds only matching copy/dump hooks and compares
all outputs, while retaining the original rain/snow checksum. Native source
initialization repair explicitly zeroes the two previously uninitialized ice
fallout accumulators, also in the CUDA header used by native HIP. Untouched
baseline snapshots remain local. The serial column recurrence in reference.cpp
is an independent CPU validation oracle, not an execution fallback. CUDA backend
source is unvalidated.

Whole native HIP/IRIS outputs agree exactly; the serial oracle independently
checks all five precipitation arrays affected by the repair. Serial th/qc/qs
thermodynamic values differ from both GPUs and remain a qualification, not a
passing full CPU comparison. No tolerance was enlarged to hide that mismatch.
