# TimeX depthwise convolution, IRIS DMem

`make run` retains B=32,C=T=768, eps=0.1, Tmax=768, BF=8, BB=2 and seed42.
`make run RUN_ARGS=--small` uses bounded B8,C3,T16; `--T 768` tests the full
kernel sequence length without the large batch/channel dimensions. Enforced
constraints remain B%8==0,T%4==0,4<=T<=768.

The original optimized forward/backward kernel bodies, shared-memory staging,
float4 vectorization, batch groups and accumulation order are retained, merely
given C-linkage IRIS entry points. Additional HIP kernels implement the original
nonlinear loss `(out*out-tanh(out)).sum()`, its upstream gradient, and the batch
sum of weight gradients. No CPU fallback; every device buffer is DMem.

CPU PyTorch conv1d/autograd independently checks every output, reduced weight
gradient, input gradient and loss (atol/rtol1e-5). `--native /path/harness`
additionally byte-compares forward/per-batch weight/input gradients against
the unchanged original HIP kernel bodies in a standalone native HIP harness.
This is kernel-level native parity, not a claim that the original framework
JIT can execute: local PyTorch is CUDA-only, not HIP-enabled. Original GPU RNG
is explicitly substituted with CPU seed42; tensors are identical for both
native kernel harness and IRIS. Production default shapes/warmup/profiler
timings are not validated or comparable to the original framework pipeline.
