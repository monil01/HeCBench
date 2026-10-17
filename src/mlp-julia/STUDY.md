> Measurement correction: the v1 Julia timing table below is superseded and excluded. The cuBLASLt default-stream launches were not covered by the task-stream synchronization used in v1. Numerical artifacts remain valid. A v2 device-wide completion protocol is being remeasured for all three Julia implementations; see `measurement_repair_scope.json`.

# MLP: three-layer FP32 inference

This study retains the source's synthetic three-layer dense network: batch 128,
input 784, hidden widths 512 and 512, output width 10. The dimensions resemble a
classifier, but there is no trained model, labeled dataset, accuracy experiment,
or documented scientific dataset. Inputs, weights, and biases follow the original
C++ seed-123 random stream and exact draw order. Four configurations independently
cover bias off/on and ReLU off/on; the source applies the epilogue to every layer,
including the final layer.

## Reference fidelity

The historical Julia file included an unrelated benchmark. It is preserved in
`reference_original/main.jl` and the CUDA.jl child's `original_main.jl`. The new
CUDA.jl reference invokes the source cuBLASLt API sequence: transpose weights,
FP32 compute/output, alpha 1, beta 0, fused epilogue, and one heuristic algorithm
selection per layer. Descriptor initialization, attribute setup, matrix layouts,
workspace preference, and heuristic selection remain inside each timed forward.

CUDA 12.0 uses stack descriptor layouts of 184/64/64 bytes for the operation,
matrix layout, and preference. The newer Julia generated bindings reserve larger
storage; their internal initializer calls explicitly pass the active native ABI
sizes. This preserves the source inline initializer operation. The source
allocates 16 MiB workspace but advertises and passes 4 MiB. Both references use
the default legacy stream, with synchronization after Julia input preparation.
The loaded cuBLASLt version/path/hash are recorded in the run and diagnostic
metadata. No opaque vendor kernel geometry is inferred from the API.

## Measurement and verification

Each process performs the source's 300 synchronized warmup forwards, followed by
five trials of ten forwards. Each trial synchronizes once at its end and reports
time per forward; the first trial is excluded from the mean. A forward includes
all three layers. Inputs and output buffers are prepared/reset outside the timer.

The original native CLI passed all four bias/ReLU cases at the original shape.
Those executions are diagnostic evidence, separate from the primary five-trial
series. Primary artifacts contain the exact 770,058-value serialized input prefix
and all 132,352 intermediate/final activations. The reserved but unused tail of
the native intermediate workspace is not read. An independent Float64 NumPy
matrix-chain oracle checks every layer using the original absolute tolerance
0.01; ReLU configurations also check nonnegativity. All five saved artifacts must
agree with both references. This verifies the named forward configurations,
not trained-model accuracy or all possible network shapes.

## Planned portable mapping boundary

KA and JACC implementations will use actual tiled FP32 device kernels rather
than forwarding to cuBLASLt. Their dense/bias/ReLU mathematics and full-forward
timed scope must match. The vendor-versus-custom reduction algorithm, tiling,
fused epilogue implementation, and vendor descriptor/heuristic overhead are
explicit confounds: timing differences cannot be attributed solely to frontend
abstraction overhead. Vendor internals remain unknown unless separately profiled.

## Completed evidence and observations

All four configurations pass across all four implementations. A separate CPU-only saved-artifact audit rechecks all 80 artifacts, every layer, exact input-prefix bytes, finite outputs and ReLU nonnegativity. CUDA and CUDA.jl outputs are byte-identical within each configuration; KA and JACC likewise share output bytes, with different reduction roundoff from the vendor pair. Maximum independent CPU absolute error ranges from 0.000573 to 0.001599 for the vendor pair and from 0.001673 to 0.004945 for the custom pair, all below 0.01. Relative errors are actually computed with denominator max(abs(oracle),0.01), not placeholders or normalized tolerance residuals.

Both targets passed their first generic kernel candidate, four configuration-check commands each, with no feedback repairs. The CUDA.jl reference required two program candidates and five checks after one harness parser rejection; this is separately costed, not global first-program success.

| Bias/ReLU | CUDA (us) | CUDA.jl (us) | KA (us) | JACC (us) |
|---|---:|---:|---:|---:|
| b0_r0 | 54.108 | 50.491 | 222.979 | 406.451 |
| b1_r0 | 58.301 | 75.035 | 225.974 | 414.826 |
| b0_r1 | 58.149 | 44.230 | 227.496 | 442.142 |
| b1_r1 | 55.760 | 41.480 | 222.634 | 406.843 |

All retained samples and variation are in the traits file. The custom/vendor gap and the KA/JACC gap are measured observations; their individual causes are not established without profiling. No fastest-run selection or causal backend ranking is made. Logical dense work is 171,180,032 FLOPs per forward under the two-FLOP multiply-add convention, plus optional 132,352 bias additions; ReLU comparisons are separate. These are algorithmic counts, not measured physical instructions/traffic or roofline positions.

## Authoritative v2 completion-boundary measurement correction

A retrospective S2 audit found that cuBLASLt launched on the legacy default stream while the Julia harness synchronized the task stream. The v1 numerical outputs remain correct, but its matched timing claim is excluded. The old source, raw results, table and source-commit receipts are retained. Device-wide blocking completion now covers preparation, all 300 warmups, pre/post timed regions and cleanup, matching native cudaDeviceSynchronize. All three Julia models were freshly evaluated for all four configurations; unchanged native source/adapter/input/clock series alone was reused under explicit cross-configuration review. No arithmetic, device kernel or vendor algorithm change was made, and no fastest-run selection occurred. The exclusion policy was recorded before new trials; an initial metadata key typo delayed sidecar application until it was caught and fixed by an effective-config audit. That T1 event and the original sidecar are retained.

| Bias/ReLU v2 | CUDA (us) | CUDA.jl (us) | KA (us) | JACC (us) |
|---|---:|---:|---:|---:|
| b0_r0 | 54.108 | 57.657 | 219.175 | 408.564 |
| b1_r0 | 58.301 | 60.228 | 219.772 | 404.281 |
| b0_r1 | 58.149 | 55.407 | 221.652 | 402.841 |
| b1_r1 | 55.760 | 64.070 | 219.719 | 404.936 |

The CPU-only replacement audit rechecks all 80 artifacts against exact input prefix and independent Float64 all-layer oracle. Original translation attempts/usage remain unchanged; this shared measurement correction has a separate prospective usage window and 12 fresh Julia validation process evaluations, not 12 new device-kernel candidates. Internal compiler/cache work is unknown. Vendor/custom algorithm and host-descriptor confounds remain.
