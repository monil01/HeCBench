# Modulated deformable-convolution custom CUDA kernels

## Certified component scope

`convolutionDeformable_full_custom_backward_triplet_4batchslices_3geometries_F32_dyadic_v1`
validates all three custom source GPU kernels and their interpolation helpers in
the source's timed backward per-batch order:

1. Offset and modulation-mask gradients.
2. Atomic accumulation of image gradients.
3. Modulated image-to-columns sampling, overwriting the old columns buffer.

The original CUDA file/header and native launch functions compile standalone,
unchanged, without a Torch shim. **Full PyTorch/DCN integration is unverified**:
the actual Torch import fails with `ModuleNotFoundError`. Offset-prediction Conv2d,
sigmoid, tensor matmuls, weight/bias-gradient integration and autograd are outside
this component contract. Upstream columns gradients are independently generated
`weightᵀ × grad_output` fixtures, outside the source custom-kernel timer. These are
legitimate device inputs, not a CPU replacement of the measured GPU algorithms.

Four per-batch slices cover three geometries: grouped channels4/2 and6/3,
3×3/2×3 kernels, stride/dilation variations and equal padding1/1 or0/0. Nonzero
±0.25/±0.375 offsets and0.25/0.5/0.75 masks exercise interpolation and derivatives.
The original100×64×128×128 workload, initial zero-offset/mask0.5 regime,
multibatch forward path and full training/application accuracy are unmeasured.

## Numerical evidence

Final receipt:
`.porting-state/portable_julia/runs/20261019T194302800649Z_convolutionDeformable/result.json`.
All20 primary captures are bitidentical:27,434 Float64-encoded values each,
including17,080 exact actual device-input/prefix values. The final columns and
all image/offset/mask gradients survive and are captured in full with guards.
The old upstream columns are captured in the input prefix before overwrite.

An independent CPU quadrilateral shape-function and adjoint evaluator checks
all values from actual inputs. Seventy-two scalar finite-difference controls
corroborate smooth analytic gradients. Tolerances were declared before native
execution: absolute2e-6/relative2e-6, exact prefixes and guards. All observed errors
are zero. The dyadic fixtures make analytic outputs exactly Float32-representable;
this does not certify arbitrary Float32 accumulation order or nonsmooth integer
sampling-knot derivative conventions. Actual GPU data, math and atomic additions
are Float32; Float64 artifact encoding and CPU controls are separate.

## Native asymmetric-padding defect

A separate actual unchanged-source PH1/PW0 diagnostic produces60/60 incorrect
image-gradient values, maximum absolute error0.5244140625. Other columns and
offset/mask gradients match. The native col2im host launcher passes `pad_h` twice.
An owned diagnostic changes only that argument to `pad_w`; all four complete
outputs then match exactly. Sixty independent image-gradient finite differences
support the expected values. Original and minimally corrected captures/logs,
source hashes and exact patch are retained in the witness.

This correction is not used in the primary equal-padding measurement; original
native source remains immutable. Target kernel code uses the original device
argument's `PW` semantics. Asymmetric target-to-original numerical agreement is
not claimed. See `asymmetric_padding_witness.json` and diagnostic source/recipe.

## Measurements and model classification

| Version | Mean of trials2–5 (µs) |
|---|---:|
| CUDA |100.109|
| CUDA.jl |144.921|
| KA |224.072|
| JACC |309.878|

Five actual checked in-process trials per version, exact all-geometry warmup
outside ROI, discard only trial1 and average2–5. Each timing sums four separate
per-batch triplet intervals:12 launches plus whole-device completion. Allocation,
input H2D, captures, CPU oracle and all Torch/library setup are excluded. Source
Torch code uses stream completion; the named adapter consistently uses
whole-device completion across all four versions. This integration-adapted scope
is excluded from untouched-native aggregates; no primary native arithmetic repair
is made. It is not full DCN or original Python-main walltime.

All three **timed custom GPU implementations** are `framework_device_kernel`:
CUDA.jl uses its own atomic API, KA uses portable Atomix, and JACC uses its exported
atomic macro. Every launch uses source512 threads/grid and0 dynamic shared bytes.
Full unverified tensor/library orchestration is not classified as framework-only.
Julia geometry argument packing and mixed-width integer index expressions differ
from native individual Int32 parameters. Host wrappers and unmeasured resources
are covariates. No roofline, occupancy, atomic-contention or causal frontend
performance claim is supported by these timings alone.

Each target passed its first program/device-body candidate and actual device-JIT
process. Three kernel entry definitions and seven helper definitions are recorded
per target. CUDA.jl creates shared device math/host capture code, inherited by KA
then JACC. Actual token windows are context/cache/session inclusive; native Torch
failure, asymmetric diagnostics, shared verification and annotation are separate
from target translation. No exclusive billing or historical recovery is claimed.

## Cold reproduction

Pinned Julia1.12.6, CUDA.jl6.2.1, KA0.9.45, JACC1.4.0 and Atomix versions/hashes
are recorded in `environment_snapshot.json`. Native nvcc12.0.140 and localCUDA12.0
run on the V100sm70. Python3.12.3 standard-library fixtures need no NumPy/Torch.
All28 generated primary fixture/parameter/plan files regenerate byte for byte.
Regenerable CPU binaries can remain local; raw GPU captures are never source Git.

```bash
python3 src/convolutionDeformable-julia/generate_assets.py
python3 src/convolutionDeformable-julia/prepare_cold_config.py /tmp/dcn-cold.json
python3 tools/verify_portable_julia.py convolutionDeformable --config /tmp/dcn-cold.json
```

The cold config relocates source/package paths, removes reuse and runs four fresh
series. The verifier owns the GPU6 lock; do not wrap it in another flock. The
CPU-only `audit_saved_outputs.py RESULT` rereads all20 captures/source bindings,
recomputes the full adjoint, regenerates the fixtures and validates the saved
asymmetric source/repair proof. It adds no GPU executions. Diagnostic CPU inputs
are regenerated by `native_diagnostics/prepare_asymmetric.py OUTPUT_DIRECTORY`.

`paper_observations.*`, `scientific_witness.json`, `domain_snippet.json`, canonical
STATE per-model traits/statuses and actual usage/calibration manifests link this
scoped result to RQ1–RQ5. Full original program and original default-regime
coverage remain explicitly unverified.
