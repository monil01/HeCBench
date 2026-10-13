Recursive Gaussian IRIS dmem port
================================

Preserves original first/simple recursive and full Gaussian kernels, float4 operations, RGBA packing, transpose geometry16x16, column group256, sigma10/order0, clamp behavior, CPU reference and tolerance. Width/height kernel scalars are size_t (64 bit), matching source kernel signatures despite uint32 host image dimensions. All buffers dmem with rw partial writes, update after input upload, and flush after each complete filter. Existing shared boxfilter-sycl shrUtils image/reference routines retained as a read-only dependency.

Native adapter adds only optional HECBENCH_DUMP of every output pixel. Validation passes deterministic P6 PPM input via existing CLI (33x29 rectangular, partial tiles; 3 cycles), compares every packed RGBA pixel byte-exact against native and checks original CPU verdict. The default StoneRGB.ppm/1000-cycle arguments remain unchanged; that external default image is not present locally. Native originals remain untouched.
