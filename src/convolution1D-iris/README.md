IRIS DMem carries the input, output, and mask. The basic, tiled, and
tiled-caching algorithms are distinct GPU kernels. Fixed shared arrays cover
the original maximum block size (1024) and halo instead of dynamic shared
allocation. All original argument and run defaults are retained.

AMD verification uses `./main 1048576 1` for both native HIP and IRIS,
covering all four masks, three precisions, five block sizes and three variants
(180 complete array reference checks per executable). Set IRIS_ARCHS=HIP.
