IRIS DMem ISO2DFD port. Original wavelet, DT/DXY, stencil arithmetic, argument
order, default 2048 2048 1000, CPU reference and 0.1 threshold are preserved.
Two DMem wavefields alternate with destination iris_rw (it reads its old time
level) and source iris_r. No host reset occurs during GPU iterations.
The original program snapshots the named next buffer even for even iteration
counts; that behavior and wavefield_snapshot.bin/wavefield_snapshot_cpu.bin
filenames are retained. Bounded odd/even tests compare the full binary fields
against native HIP in isolated run directories, in addition to native PASS.
Dimensions under 24 are rejected to avoid unsigned wavelet-init underflow.
Intel MIT attribution retained. Nonfinite values fail reference validation.
