# Deformable-convolution observations

Configuration `convolutionDeformable_full_custom_backward_triplet_4batchslices_3geometries_F32_dyadic_v1`. **Custom CUDA-extension kernel scope only; full PyTorch/DCN integration is unverified.**

## RQ1: fidelity and scientific guarantees

All20 complete27,434-value captures are bitidentical, including17,080 actual input/prefix values. Full columns and image/offset/mask gradients match an independent shape-function/adjoint oracle, with72 smooth finite-difference controls. Meaningful nonzero offsets and masks exercise interpolation and gradient scatter. Dyadic fixtures produce zero observed error within predeclared2e-6 absolute/relative budgets.

Actual asymmetric-padding native launch passes pad_h twice: all60 image-gradient values fail, maximum error0.5244140625. A minimal owned host-argument change restores every output exactly;60 input-gradient finite differences support the oracle. This diagnostic correction is not used in primary equal-padding timings.

## RQ2: feedback and tokens

All three targets pass first candidates/device-JIT processes. Three GPU entry definitions and seven helper definitions per target. CUDA.jl creates shared math/host capture code; KA then JACC reuse it. Context/cache/session-inclusive actual usage windows remain separate from native preparation, unavailable Torch integration, asymmetric diagnostics and shared audits.

## RQ3: execution

| Version | Mean2–5 (µs) |
|---|---:|
| cuda | 100.109 |
| cuda_jl | 144.921 |
| ka | 224.072 |
| jacc | 309.878 |

Sum of four backward per-batch triplet intervals (12 launches) plus whole-device completion; allocations/transfers/checks/Torch matmuls/offset prediction/autograd excluded. Upstream columns gradients are independently generated W^T grad_output fixtures. Small named shapes, wrapper/geometry ABI and integer-index width are covariates.

## RQ4: mechanisms; RQ5: selection

Actual Float32 atomic scatter uses each framework API: CUDA.jl own atomics, KA Atomix, JACC exported atomic macro. No foreign CUDA device extension is required for these named kernels. This does not establish tensor-library/autograd integration, original100-batch throughput or training accuracy. No measured roofline/occupancy/atomic-contention claim. The default zero-offset/mask.5 and multibatch forward paths are unmeasured; gradient controls cover smooth noninteger offsets.

Complete source hashes, finite-difference observations, scope limitations and usage windows are in JSON records.
