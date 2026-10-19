# Braycurtis: full source distance and gradient kernels

This is a **native kernel-only integration adapter**. Local Python cannot import PyTorch (actual ModuleNotFoundError retained in native_diagnostics/torch_import.log). `extract_kernels.py` extracts the exact four original CUDA device-function byte spans; `extraction_manifest.json` records source/span hashes. `accessor_shim.h` supplies reviewed contiguous row-major size/stride/indexing, source-equivalent grid-stride looping and finite signed signum. The source device arithmetic is unchanged. The original tree is immutable; MIT attribution is preserved in LICENSE.source.

The original PyTorch/autograd registration, contiguous-copy/dispatcher overhead and Python test CLI are **unverified**. This result is excluded from untouched-native aggregates. It is an integration adaptation, not a mathematical native reference correction or a framework support barrier. No ATen ABI equivalence is claimed.

## Scientific contract

Compute D=sum(abs(x-y)), S=sum(abs(x+y)), d=D/S and both input gradients of sum(grad_output*d). All four kernels are ported: distance, components D/S, gradient_x1, gradient_x2. Every final distance, both component arrays and both gradients survive and are captured with guards.

Float32 and Float64 each use (B,M,N,K)=(1,3,5,7),(2,17,11,33),(1,65,67,129). The small fixture is strictly positive; larger fixtures contain both signs. Every per-feature difference/sum is away from the absolute-value cusp and each pair denominator is positive. Source Float16 dispatch and the original much larger Python test geometries are unmeasured, not unsupported. Zero denominator is mathematically undefined; the source's zero-sign subgradient policy is outside this smooth-domain contract. The [0,1] distance bound is checked only for the nonnegative fixture.

Each trial captures162,154 finite Float64-encoded values, including95,346 exact actual-device input-prefix values. Encoding Float32 input values as Float64 is lossless; artifact precision is not compute precision. Complete input buffers are read before/after execution, actual dimensions are echoed by an untimed device observer, and every input/output guard is checked. Each trial has66,328 final algorithm values plus guard/metadata/input values. Not all overwritten R10 intermediate launches are captured.

All five trials/model pass both references and independent checks. Julia captures are bitidentical; native CUDA exhibits tolerated floating rounding differences. Fieldwise CPU acceptance was declared before execution: Float32 atol1e-5+rtol1e-4; Float64 atol1e-12+rtol1e-10. Prefix and guards are exact. The primary runner uses the global Float32 dual budget while each model enforces stricter fieldwise CPU budgets. An additional saved-series audit also directly checks all25 ordered trial pairs under those per-precision field budgets, including strict F64: all pass with no GPU retiming.

Independent checks include Python math.fsum forward/components and algebraically expressed quotient gradients, central finite differences of all56 small positive-case input partials, a NumPy vectorized oracle reconstructed from actual device-input readbacks (no oracle asset read), and all1,848 signed-case scalar partial finite differences (max1.77e-9, control budget1e-7). Nearzero gradients can have a relative error above rtol while passing the absolute-plus-relative criterion; no uniform relative-only claim is made.

## Regeneration and measurement

The essential CPU binaries, generated fixtures and hashes are listed in ready_support_files.json/study_plan.json. Python3.12.3 standard library only:

```sh
python3 src/braycurtis-julia/generate_assets.py /tmp/braycurtis-regenerated
```

Actual isolated cold regeneration reproduced all23 files exactly; compare with generation_reproduction.json before copying reconstructed assets into missing locations. `extract_kernels.py` regenerates the exact source extraction, but preserve measured files when they are already present. The extraction spans can be revalidated without rewriting source using the hashes in extraction_manifest.json. Original source includes, extracted include, shim, plan and runtime assets are all explicitly bound by the verifier.

Fresh reproduction from the source repository root:

```sh
python3 src/braycurtis-julia/prepare_cold_config.py --output /tmp/braycurtis-cold-config.json
python3 tools/verify_portable_julia.py braycurtis --config /tmp/braycurtis-cold-config.json
```

This relocates absolute paths and removes all saved-series reuse. The verifier builds native code in isolation, runs physical GPU6 under its internal shared lock, warms the complete forward/backward host loops for all shapes/precisions, and executes five checked trials/model. Do not wrap the verifier in an outer flock. Discard only trial1 time and average2–5.

The reported time is the **sum over all six cases of a forward R10 mean and a three-kernel backward R10 mean**. Whole-device completion separates forward/backward regions and ends each batch. Allocations/transfers, resets, observer, CPU checks and serialization are outside ROI. This is a mixed-precision/mixed-size composite, not a per-precision throughput, original PyTorch wrapper timing or whole application time. All raw five timings remain retained.

Pinned projects src/_julia_ka_env and src/_julia_jacc_env are unchanged from the actual environment snapshot at nn-julia/environment_snapshot.json: Julia1.12.6/CUDA.jl6.2.1/KA0.9.45/JACC1.4.0/localCUDA12.0/V100sm70. Native nvcc12.0.140, stdc++17/O3/sm70. Native legacy-default-stream launches and Julia task-stream launches share whole-device completion boundaries; wrapper/accessor/resource differences are disclosed, no causal performance explanation or roofline coordinate is inferred.

CPU-only audits while saved captures remain available:

```sh
PYTHONPATH=/tmp/pasc-julia-plot-tools python3 src/braycurtis-julia/audit_saved_outputs.py
PYTHONPATH=/tmp/pasc-julia-plot-tools python3 src/braycurtis-julia/check_signed_finite_differences.py
python3 src/braycurtis-julia/audit_per_field_dual.py
```

NumPy2.5.4 is used only by supplemental CPU audits, not the generator. Repeated mathematical calculations may be cached only after every artifact/prefix is fully rehashed; each actual GPU trial/timing remains separate. Actual GPU captures stay local and are not Git assets; their hashes cannot recover missing data.

## Mapping and cost

All three Julia versions use framework_device_kernel: ordinary scalar abs/sign/division and own framework global indexing,256 threads/group, no authored shared tile, atomics or foreign CUDA device intrinsics. CUDA backend allocation/copy/completion is host harness work. Native packed accessors become explicit row-major flattened Julia views; no implicit column-major transposition is used.

Each target passed candidate1/device-JIT-process-check1 with four algorithm definitions and one untimed parameter observer. Two precisions/three shapes and five timing trials are distinct from translation attempts. CUDA.jl authored shared device/host helpers; KA then JACC reused them. Actual input/cached/output/reasoning windows and calibration strata preserve inherited session/cache/context and samebenchmark coordination. One token-metadata wrong-key failure, before writes/device compilation, is shared tooling effort; it is not a target code failure. These counters are neither exclusive target billing nor recovery of historical consumption. See paper_observations.json for RQ1–5 and limitations.
