# TGVNN: complex dynamic-MRI reconstruction

## Certified contract

The primary configuration uses the original tracked pincat image/mask, 128×128×50 complex samples, 500 source primal-dual iterations and original defaults (mu=1). The second configuration is a deterministic rectangular 7×5×4 complex fixture with seven iterations. Original native CLI diagnostics passed both one and 500 iterations; these single diagnostics are distinct from the selected five-trial study.

All 27 original GPU definitions are represented by source-derived per-index Julia methods. CUDA.jl, KA and JACC supply their own entry or callback. The shared host C ABI contains only runtime allocation/copy, cuFFT, cuSOLVER, cuBLAS and CPU scheduling; it does not delegate authored GPU kernels to CUDA. The complete timed implementation is `hybrid_host_control`, with `framework_device_kernel` components, for each Julia model. It remains CUDA-library-specific despite portable pointwise APIs.

Native GPU arithmetic is unchanged. The named native adapter controls streams, observes defined states, adds caller guards and excludes printing/file IO. This is integration-adapted, not a mathematical native correction; exclude it from aggregates requiring the untouched native CLI.

## Evidence and performance

Final primary: `20261020T060153233158Z_tgvnn`; control: `20261020T060045696702Z_tgvnn`. Canonical receipts are under `.porting-state/portable_julia/runs/`; physical new artifacts are under `/tmp/hecbench-portable-infrastructure-state`. Storage provenance preserves the aliases and hashes. Preserve these paths before temporary-storage cleanup.

Each model/configuration has a completed warmup and five checked trials; reported mean excludes trial1. Primary seconds: CUDA 4.0595610625, CUDA.jl 4.21340137725, KA 4.25855849475, JACC 4.27850763175. Control: .009884108, .010310956, .01286116375, .0154381655. The ROI is the completed full source `tgv_cs` pipeline including plan/workspace creation, H2D, iterations, GPU quality computations and source cleanup. Observer copies/checks and outer allocation/IO are excluded. Whole-device fences cover legacy-stream library calls.

All40 selected artifacts total5,768,177,760B. Every actual input prefix is byte-checked, followed by all defined surviving42N complex states, SER/singular values and caller guards. Independent Complex128 finite-difference/FFT/Gram-eigen nuclear-proximal evaluation uses the predeclared per-field engineering budget `1e-6 + .002*max_abs_component(field)`; GPU dual-reference tolerance is atol=rtol=.0002. These are finite-value numerical agreements, not bitidentity. Eight actual untimed third-iteration traces contain80 nonzero intermediate fields (839,004,160B). Raw nonunique SVD factors are retained and checked for orthogonality and reconstruction, without requiring identical phases. No claim covers every overwritten state in all500 steps.

`final_source_binding_audit.json` rehashes115 bound source/package/asset files and eight unchanged source trees. `environment_snapshot.json` separately records Project/Manifest/preferences: these environment files were manually snapshotted, not automatically part of the original measured dependency bindings. `vendor_resolution.json` records actual common FFT/BLAS/solver DSOs. Native static and bridge dynamic cudart12.0 instances differ. Julia wrapper allocations/GC/dispatch and vendor opaque internals remain covariates; no profiling, roofline position or causal frontend claim is inferred from elapsed time.

## Cold CPU assets

Essential original RA files are already tracked under `src/tgvnn-cuda/data`. Generated CPU assets (~1.43GB), host `.so`, and measured GPU captures stay local; commit recipes, requirements and exact hash manifests rather than large generated binaries. No irreplaceable new external input exists.

With the pinned NumPy2.5.4 environment and existing CUDA headers/libraries:

```sh
PYTHONPATH=/tmp/pasc-julia-plot-tools python3 src/tgvnn-julia/regenerate_assets.py --output /tmp/tgvnn-assets --proof /tmp/tgvnn-assets-proof.json
python3 src/tgvnn-julia/prepare_cold_configs.py --assets /tmp/tgvnn-assets --output /tmp/tgvnn-configs --capture-root /tmp/tgvnn-new-captures
STATE=/tmp/tgvnn-new-state PYTHONPATH=/tmp/pasc-julia-plot-tools python3 tools/verify_portable_julia.py tgvnn --config /tmp/tgvnn-configs/pincat.json
```

Use central verifier directly for relocated configs (the convenience `run_study.py` requires configs inside this benchmark directory). Its own global GPU lock must not be wrapped in another flock. CPU regeneration uses eight BLAS threads for evolution and original library-default/unset thread variables for scalar SER completion. `cold_regeneration_proof.json` proves all92 binary assets (1,428,930,576B) and host library exact on this environment. The first cold attempt exposed two scalar norm-reduction last-bit differences; its log and environment correction remain. The final proof resumed the completed independently generated primary arrays, then rechecked all hashes. Fresh relocated GPU reproduction was prepared but not executed.

## Scientific and effort limits

Source SER compares magnitudes, not complex reconstruction error or clinical quality. The selected mu1 reconstruction is checked; a separate CPU-only mu2 temporal-adjoint endpoint identity witness is out of scope and is not an observed GPU failure. Alternate datasets, parameters and general convergence/optimality are unmeasured.

Each target submitted one authored candidate bundle, two actual device-JIT process checks (two configurations), and zero target feedback repairs. CUDA.jl created27 common math methods and the host controller/C ABI; KA/JACC reused them. One JACC callback uses one framework-provided entry; it is not an authored entry template. Source-level37515 primary custom launches are not compiler attempts. Shared native-header, independent-oracle-axis, capture-schema/lifetime, premature cold preparation and scalar-generation environment failures are retained separately. One earlier self-checking native series lost raw captures; it is not promoted to artifact-verified evidence or reconstructed.

`observed_tokens.json`, calibration records and `usage_scope_review.json` describe context/cache-inclusive phase counters, inherited planning and shared-harness reuse. Counts are not exclusive billing or recovered historical cost. See `scientific_witness.json`, `paper_observations.*` and the support manifest for exact scope and source hashes.
