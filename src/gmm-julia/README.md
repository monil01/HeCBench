# GMM: full adaptive Gaussian-mixture pipeline

This named experiment uses the original DVC dataset and the full source EM/convergence/model-order pipeline, starting with six clusters and considering orders down to one. The original CSV reader skips its first line and truncates for alignment: 992 of the original1,000 two-dimensional points are used. It selects three clusters after44 EM updates (orders6/5/4/3/2/1 use21/2/15/2/2/2 updates), executing278 calls across all seven selected GPU entrypoints. The source’s unused `mstep_covariance1` is not counted as translated. The original default500-cluster CLI is not certified by this study.

## Numerical and scientific scope

All20 saved artifacts are byte-identical across the four measured versions. Each contains6,839,533 Float64-encoded fields, including16,299 exact actual-device input/initial-state/guard fields and all guarded pre/post-stage state buffers, likelihoods, metadata and the saved best model. The independent CPU audit reconstructs input bits from the original dataset and validates every stage, guard and selected model copy.

The independent Float64 mathematical gate makes508,138 checks per trial: Gaussian quadratic forms, stable responsibility normalization, weighted counts/means/covariances and analytic2×2 inverses/normalizers. Declared engineering budgets and exact input/guard policies are in `study_contract.json`. A symmetric one-Gaussian analytic control passes50 checks; a deliberately incorrect half-responsibility control is rejected. These checks establish the named operator contract, not a global optimum for nonconvex mixture fitting, statistical ground truth, or validity on other datasets. Source host merging uses a base10 determinant convention while GPU Gaussian likelihoods use natural logs; that legacy heuristic is retained and disclosed.

## Qualified native reference

Original sources remain intact in `../gmm-cuda` and `reference_history`. The original five-cluster native control passes its CPU checker but compute-sanitizer reports166,656 uninitialized global reads in unused tail covariance terms. Membership storage is explicitly padded, so this is not an out-of-bounds claim, nor evidence of an observed active-output error. Guarding each of six inactive-cluster accumulation terms removes these unused reads; the corrected probe reports zero sanitizer errors and identical CPU-checker maxima. Full provenance, source hashes and scientific logs are retained. The study additionally declares zero-initialized guarded allocation payloads. These are conditional corrected-reference results, excluded from untouched-native aggregates.

## Implementation boundaries

All versions use the original compiled C CPU controller for convergence, host normalization, model scoring/merging and allocation. The owned launch adapter changes only ten launch sites to the same callback ABI; all other controller statements are preserved. Each Julia version authors the seven actual GPU operations. A nonnull Julia callback takes the GPU dispatch branch; no native GPU implementation is called by a target. Julia uses non-owning typed views over the common C-owned CUDA allocations. This is explicitly a hybrid host pipeline, not a pure whole-program Julia rewrite.

KA uses framework local memory and synchronization. JACC preserves the source shared-memory reductions through CUDA backend extensions, with explicit complete stage/model-order launch shapes and zero dynamic shared memory. Its static shared memory remains nonzero. Resource declarations and host boundaries are recorded separately per model; no compiled resource counters or roofline attribution were measured.

## Reproduce

From the repository root with Python3, unprivileged CUDA12/C++17 tools and the pinned Julia projects:

```sh
python3 src/gmm-julia/regenerate.py
python3 src/gmm-julia/make_config.py --output /tmp/gmm_fourway.json
python3 tools/verify_portable_julia.py --config /tmp/gmm_fourway.json
```

`regenerate.py` recovers the declared6,282-byte data archive, verifies MD5/SHA256 and its33,000-byte data file hash, and extracts only validated regular data. It never executes downloaded code. It was independently exercised into `/tmp/gmm_asset_regeneration_review`. The central verifier owns the GPU6 lock: do not nest another lock around it.

To repeat the CPU-only capture audit with retained local evidence:

```sh
python3 src/gmm-julia/audit_captures.py --result RESULT_JSON --output AUDIT_JSON
```

`prepare_controller.py` reproduces the committed launch-site adapter without generating acceptance data. The mathematical CPU control can be built with `g++ -O2 -std=c++17 src/gmm-julia/probe_independent_oracle.cpp -o /tmp/gmm_oracle`; normal execution succeeds and an extra argument deliberately produces exit8 by injecting wrong responsibilities.

## Timing and effort

All models complete one full pipeline warmup, then five independently checked trials. Only the first timing is excluded; average trials2–5. Mean seconds: native0.00746226625; CUDA.jl0.01126749425; KA0.01237808875; JACC0.02114141175. The ROI includes the full source CPU controller, allocation/zeroing/transfers, convergence/model-order/merges, GPU dispatch and completion. Explicit pre/post-stage observations and independent checks are excluded by recorded observation intervals; raw whole-controller and excluded durations are retained in every trial. These segmented wall times are not kernel-only or unsegmented original CLI times.

All three targets passed their first authored program and device-JIT candidate. Native/sanitizer preparation, one verifier asset-path setup failure, and one CPU analytic control fixture repair are separate units. `usage_summary.json`, per-model accounting and calibration files preserve actual prospective counters, cached/input/output/reasoning subsets, model metadata and context reuse. KA/JACC reuse the accepted CUDA.jl helpers and callback harness: costs describe this ordered marginal workflow, not independent cold-start framework costs.
