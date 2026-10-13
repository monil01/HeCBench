LDA IRIS DMem preserves seed123, topics1000, words10266, blocks500x256,
64 E-step iterations, repeat argument and default100. Shared working memory is
statically sized to unchanged4000 floats. Logical warp32 reduction retained as
in HIP sibling. All device intermediates iris_rw; validation mask refresh uses
DMem.update(host), losses flush before summation. Numeric baseline is both finite
train/validation losses, rtol1e-5 atol1e-4. CUDA backend source unvalidated.
Modified host/kernel launch and memory ownership for IRIS; original license retained.
