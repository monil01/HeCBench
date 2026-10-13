BM3D retains the original seven active device kernels, fixed 8x8 patches,
16-member groups, search parameters, batch shape, color conversion, sigma=20
default invocation and reference PSNR calculation. All buffers use DMem,
read/write access for partial writes and in-place transforms, host updates for
aggregation resets, and a final output flush.

P1: the two dynamic workspaces are static arrays with exactly the original
default sizes (38504 bytes for matching and 4352 bytes for hard thresholding).
The original logical warp is 32 lanes; HIP shuffles explicitly select width 32.
The native HIP sibling omitted that width despite retaining warpSize=32;
an isolated native adapter restores the CUDA width without editing originals.
No transform or denoising algorithm is replaced.

REPEAT defaults to 100. Explicit bounded verification builds can override it
through -DREPEAT=1 or 2 while production source defaults stay intact. The same
guard is applied only in the isolated native adapter, which also adds opt-in
BM3D_VERIFY raw pixel artifacts. Original full 1024x768 input/reference images
and all other defaults are preserved. Scratch padding is zero-backed but is
not used by the original logical pixel operations. CImg is the unchanged
sibling CPU image-I/O header; ImageMagick provides PNG loading/saving.

Numerical comparison covers every RGB component and original reference PSNR.
Native HIP repeats vary by up to two integer pixel levels due to float atomic
aggregation and integer color conversion. Acceptance requires max difference
<=2, mean difference <=0.001, and PSNR difference <=0.001 dB. The failed
stricter max-one-level check and native reproducibility evidence are retained.
