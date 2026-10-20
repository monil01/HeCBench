# SA: genomic suffix-array construction

The study ports the original DC3 suffix-array algorithm. All nine custom GPU definitions and the data-dependent recursive host controller are implemented in CUDA.jl, KA and JACC. The common C ABI exposes only the source's Thrust device containers, `sort_by_key` and `exclusive_scan`; it contains no DC3 controller or custom benchmark kernel. All queues use legacy-stream semantics and explicit whole-device completion.

## Verified scope

The primary case uses the first **1,000,000 ASCII symbols of the original genome dataset** and the source's **100 constructions** per timing sample. Fourteen separate R1 controls cover unique, uniform and periodic strings plus real prefixes, including all three length remainders. Every model has five checked trials; the reported mean uses trials 2–5. Control timings are diagnostic and excluded from aggregates.

All saved input words, full suffix indices, three source zero sentinels and seventeen external guard words agree exactly across all four models. The independent complete certificate validates the permutation and strict adjacent `(first symbol, inverse rank of tail)` order, with empty-tail rank −1. Finite-tail descent proves the unique full lexicographic order. It is not another DC3 implementation or the source's weak checksum.

Only the final construction of each internal repeat sequence is archived and checked per outer trial. Private Thrust workspace guards and all internal intermediate arrays are not claimed. This establishes indexing correctness for the named inputs, not biological alignment or downstream application quality.

## Performance boundary and qualifications

The ROI includes input/prior-suffix H2D copies, recursive DC3, Thrust allocation/sorting/scanning, GPU-rank readback, dependent host recursion, workspace destruction and final D2H output. It ends with whole-device completion. Warmup, actual device-input readback, certificates and serialization are outside. Time is divided by the source repeat count.

Primary mean construction times from runs 2–5: native CUDA **29.1206 ms**, CUDA.jl **28.8980 ms**, KA **29.0130 ms**, JACC **30.8657 ms**. These are complete mixed pipeline times. FFI, nonowning array wrappers, host dispatch/GC and compiler paths are covariates; no causal overhead attribution or roofline positioning was measured.

The custom device component of each implementation is `framework_device_kernel`. The complete measured pipeline is `hybrid_host_control`, because an actual device-derived rank controls recursion, with Thrust library operations. CUDA.jl host pointer wrapping is not a foreign device intrinsic. KA and JACC custom entries use their own kernel APIs. JACC declares zero dynamic shared memory and retains the source 256-thread geometry. Thrust's internal resource use is opaque here.

Original CUDA mathematics and kernel bodies are unchanged. Guarded external storage, warm protocol and shared-runtime linking are integration adaptations, **not a mathematical native repair**, and are excluded from untouched-native aggregates. The original unchanged 1M/R1 CLI was separately executed; its capture-only variant passed the full independent certificate. Actual R100 study trials use the documented owned harness. Only positive ASCII and the pinned Thrust backend are certified; no claim is made about other tie-order policies/backends or arbitrary binary alphabets.

## Receipts and effort

- Primary final receipt: `.porting-state/portable_julia/runs/20261020T032722142031Z_sa/result.json`.
- Control final receipt: `.porting-state/portable_julia/runs/20261020T032648753910Z_sa/result.json`.
- `saved_audit_final_{primary,controls}.json`: all forty complete captures, 160,903,840 bytes, rehashed and independently certified. Raw GPU captures remain local, outside source Git.
- `final_source_binding_audit.json`: exact original/canonical source-tree digests and every explicit bound dependency unchanged.
- `study_adjudications.json`: hash-bound descriptive scope and independent-audit links; no acceptance or timing changes.
- `scientific_witness.json`, `domain_snippet.json`, `environment_snapshot.json`, `reference_effort_537.json`, `paper_observations.{json,md}` and `observed_token_windows.json`: scientific, implementation, effort and usage evidence.

One GPU math source bundle and nine custom entries were authored per target. CUDA.jl required a host `Ptr` → `UInt` → `CuPtr` API correction before custom device JIT. JACC required one generator-whitespace parse fix before JIT. KA's first program passed. Both successful configurations execute in distinct Julia processes; actual backend compiler invocation totals are unknown. A separate initial runner runtime-asset setup failure never launched a target process. All failed sources/logs and source-unit definitions are retained. Token windows include inherited context/cache activity and are not exclusive billed costs or historical recovery; calibration eligibility is independently adjudicated.

## Cold reproduction

The original anonymous DVC object is 1,251,585 bytes with MD5 `8cd52d0395bc0f76df02e438a4fa673d` and SHA256 `c2dffd66e25531d44415c52db57461d164aee6e6bcfff7254cfd78726096351a`. The fetcher validates size and hashes before safe extraction. The 4,639,676-byte genome generates every study input deterministically. No irreplaceable binary fixture is required. `study_plan.json` records exact fixture bytes/hashes. Eighteen generated inputs/plan/case-list files were regenerated in fresh directories and byte matched; see `cold_regeneration_proof.json`.

From the source repository root, using the pinned CUDA12.0 toolkit, Julia1.12.6 projects and GPU6:

```bash
python3 src/sa-julia/regenerate_assets.py src/sa-julia
/usr/bin/nvcc -O3 -std=c++17 -arch=sm_70 --cudart shared -Xcompiler -fPIC -shared src/sa-julia/thrust_bridge.cu -o src/sa-julia/libsa_thrust_bridge.so
python3 src/sa-julia/prepare_cold_configs.py --source-root "$PWD" --output /tmp/sa-cold-configs
PYTHONPATH=/tmp/pasc-julia-plot-tools python3 tools/verify_portable_julia.py sa --config /tmp/sa-cold-configs/sa_cold_primary.json
PYTHONPATH=/tmp/pasc-julia-plot-tools python3 tools/verify_portable_julia.py sa --config /tmp/sa-cold-configs/sa_cold_controls.json
```

The runner owns the GPU lock; do not wrap it in another flock. The cold configurations contain no reused results and bind regenerated DLL/input bytes. An existing exact archive can be supplied with `--native-assets /path/to/cache`. Input generation uses Python standard libraries; the independent saved-array audit uses NumPy2.5.4. The shown `/tmp` Python dependency path is the study's existing user-space installation, not an OS package installation. No privileged command is needed.

Current-root configuration preparation and CPU regeneration were executed. A fresh relocated four-model GPU reproduction was **not** executed during annotation; the commands describe that separate reproduction. Generated shared libraries and CPU inputs are local/rebuildable; raw measured GPU arrays are not regenerated evidence.

## Attribution

Original `sa-cuda` source headers attribute Gangliao (2014), all rights reserved. No SA-specific license file was present; another benchmark's license is not assigned. Original source files remain unchanged.
