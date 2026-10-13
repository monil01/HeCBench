IRIS DMem Hogbom deconvolution port. Preserves file/iteration arguments,
default input paths and1000 iterations, gain0.1, threshold1e-5, peak reduction,
PSF subtraction and enabled CPU model/residual reference checks at atol1e-3.

The two kernels use typed IRIS dispatch. Mutable residual data remain in DMem;
peaks and final residual flush before host use. No native host allocation/copy
or kernel launch remains. IRIS finalizes before main returns.

Validated three iterations against native HIP. The existing local default-named
input files contain32x32 float images, not4096x4096. Both full model/residual
checks PASS; evidence describes this bounded input size explicitly.
