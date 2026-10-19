# clenergy Julia variants

Native original cenergy body is included unchanged in study_native.cu. CUDA.jl, KernelAbstractions and JACC each implement the charge-grid kernel in their own folder. Fixtures are generated deterministically in study_native.cu and study_common.jl; no external binary dataset is required.

From repository root:

```sh
python3 tools/verify_portable_julia.py --config src/clenergy-julia/study_config.json
```

The verifier serializes GPU6 and performs five trials per model. All five correctness checks count; timings average trials2–5. See contract.json for geometry, charges, tolerances and exact prefix/output sizes. Each21910-value Float32 artifact contains19350 actual input/guard prefix values and2560 computed grid outputs. Full independent Float64 direct sums and exact controls are checked locally; the verifier also compares each target with both GPU references.

Timing is the sum of six kernel launch/completion fragments across five named fixtures, with preprocessing, transfers and observations outside ROI. This does not measure the original default workload or whole CLI. Native approximate rsqrt and constant-memory path versus Julia generic inverse-square-root and array path are recorded covariates. See paper_observations.md/json for scientific use, compiler attempts and token scope.

Completion API detail: native uses cudaDeviceSynchronize; Julia uses current-task-stream CUDA.synchronize(). All measured kernels in these harnesses use that stream. Different host wait APIs remain a timing covariate; see completion_scope_review.json. Earlier context-completion wording does not mean identical API calls.
