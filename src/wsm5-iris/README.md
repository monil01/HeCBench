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

Whole native HIP/IRIS outputs agree exactly. The original CPU validation
attempts03/04 passed only the five precipitation arrays: th/qc/qs differed
because unqualified abs(float) resolved to host abs(int). In particular,
integer-truncated rates prematurely triggered the saturation guard and skipped
snow deposition. Those original failed-oracle sources and artifacts remain
preserved; they are not retroactively classified as full-oracle passes.

Follow-up attempt05 adds only `using std::abs` in reference.cpp so the shared
column equations use floating-point overloads, as in HIP. No GPU/native kernel,
input, recurrence, tolerance or stopping rule changed. The corrected CPU run
checks all11 arrays against both existing native HIP/IRIS repeat2 artifacts at
the original rtol1e-5/atol1e-6. All six volume fields match exactly; maximum
precipitation difference is4.440892e-16. No new GPU execution was required.

Coverage is original60x46x3, timestep10, uniform0.001 input and two reset
repetitions; default repeat10 and other physical inputs were not measured.
This is full CPU/GPU equation-execution agreement on that case, not independent
physical-model or observational validation: the CPU harness shares kernel.h.
Build the validation-only CPU oracle with
`g++ -O3 -std=c++17 -I. reference.cpp -o reference`, then `./reference out.bin`.
