# ans IRIS port status

Status: unverified, numeric failure.

This directory contains an IRIS dmem-only translation attempt for the `ans`
benchmark. Native HIP is the validation reference. With `RUN_ARGS=1`, native
HIP completes the lambda sweep without mismatch lines. The IRIS HIP run builds
and executes, but still reports a deterministic mismatch at lambda `1.38`.

The broad decoder path is functional after replacing HIP device allocations and
copies with `iris::DMem` and explicit phase flushes, but the multi-phase sync
metadata path still diverges for that one case. Do not count this benchmark as
a validated IRIS port until the remaining numeric mismatch is fixed.
