# resize IRIS port status

Status: unverified.

The native HIP baseline builds and runs, but it emits only timing/performance
lines. It does not print `PASS`, numeric summaries, checksums, or output
artifacts that can be compared against an IRIS run. Per the IRIS porting
runbook, timing-only output is not correctness evidence, so this benchmark is
left unverified until a comparable HIP-side correctness signal is available.
