The six original device kernel algorithms and software mirror-addressed
bilinear sampler are retained. Texture descriptors are unpacked into a DMem
argument and three int scalars, then reconstructed inside the device kernel.
The Jacobi template retains its original 32x6 specialization and block barrier.
No library solver or replacement algorithm is introduced.

The original five levels, three warps, 500 solver iterations, alpha=0.2,
stride alignment 256, CPU reference, L1 threshold 0.05, input filenames and
FlowCPU.flo/FlowGPU.flo outputs are preserved. Buffers use DMem only, with
read/write access for partial writes, update for host resets, pointer ownership
swaps for Jacobi/pyramid ping-pong, and a final output flush. Unused scratch
padding is zero-backed; kernels preserve the original logical pixel behavior.
No RNG is used. Native HIP validation uses unchanged source in isolated build
and run directories, with only its Common include path supplied explicitly.
