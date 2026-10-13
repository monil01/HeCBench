IRIS DMem LRN port. Seed123, N=6/5, C=150, default spatial100x160x160,
repeat10, normalization constants, 64/128/256/512 workgroup sweep and original
checksums are preserved. HECBENCH_LRN_D/H/W optionally bound spatial sizes.
HECBENCH_DUMP optionally saves full forward, backward-dst and gradient buffers.
The source data_off omits depth, backward get_omega accumulates into an integer,
and its checksum reports the unchanged dst buffer. These inherited behaviors
are preserved, not treated as an implementation of corrected general LRN.
Forward dst and backward diff_src use iris_rw because only a subset is written;
their original zero/source initialization preserves untouched entries. Backward
dst is an iris_r buffer initialized from src. Actual gradient is also flushed
for external validation. Native validation adapter adds matching optional shape
and dump controls only, retaining kernels/defaults. Full default tensors are
large (up to23GB GPU memory); bounded validation uses original N/C at D2,H3,W5
and D3,H4,W7, comparing every tensor element against native HIP.
