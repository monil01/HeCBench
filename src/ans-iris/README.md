# ans IRIS port status

Status: validated on AMD MI100, HIP-only IRIS DMem.

The full multi-phase ANS decoder uses DMem/tasks; encoding and the original
whole-input roundtrip oracle remain on the CPU as in CUDA/HIP. Native HIP and
IRIS pass all 15 lambda values at both 1 MB and the original 10 MB default.

The original CUDA/HIP decoder terminated when compressed bits were exhausted,
but a final symbol can consume no bits. The last subsequence now finishes by
its declared output count, decoding from the ANS state/table on the GPU. No
output is copied from the original data and no zero-fill hides the omission.
This minimal repair is applied identically to CUDA/HIP/IRIS, retaining the
parallel multi-phase algorithm. Previous sources and attempts are preserved.

Set `HECBENCH_VALIDATE_SENTINEL=1` to prefill output with 165 and
`HECBENCH_DUMP=path` to append complete decoded streams for each lambda. Both
sizes pass every original-input byte check with sentinel prefills; all
173,015,040 decoded bytes match native HIP/IRIS exactly across the two runs.
The host prefix scan remains as in native HIP; HIP GPU decoding never falls
back to a CPU decoder. CUDA kernel execution was not available locally.
