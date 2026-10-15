# Complete DCNv2 workload with IRIS DMem

`make run` keeps source shape (100,64,128,128), 64 outputs, 3x3/pad1,
two deformable groups, seed 0 CPU input/main weight creation and zero offset
predictor initialization. `RUN_ARGS=--small` bounds the validation fixture;
`RUN_ARGS='--small --offset-fixture'` additionally exercises learned nonzero
offsets/masks. No production-default validation/performance claim is made.

HIP tasks implement ordinary offset/mask convolution, sigmoid modulated
bilinear deformable convolution, mean target-minus-output loss, all five DCN
gradients, and backward through the offset/mask predictor (including its
parameter gradients). The original input does not require gradients; this
port additionally validates the mathematically defined input gradient.
All buffers are DMem. CPU PyTorch supplies fixtures and an independent
differentiable gather/einsum oracle, never benchmark computation.

Scope-expanded portability compromise: original ATen matmuls/cuDNN dispatch
are replaced by direct HIP math and atomic gradient accumulation, not optimized
vendor kernels. Timings/performance differ substantially. All output, mean,
input/main-weight/bias gradients, pre-sigmoid predictor output gradients and
predictor weight/bias gradients are compared, atol=2e-6/rtol=5e-5 (FP32
accumulation-order differences). Both default initialization and nonzero-offset
fixtures are tested. The target is explicitly CPU uniform rather than original
GPU uniform because local PyTorch is CUDA-only, with no available CUDA GPU;
loss gradients are independent of target. No native HIP parity is claimed.
