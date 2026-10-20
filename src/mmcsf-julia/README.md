# MM-CSF all-mode sparse tensor contraction

**Original strict Float32 primary: blocked. Qualified mixed-precision secondary: all four models verified.** This is MTTKRP, not a CP/ALS solver. The original GPU contribution statements are retained, but the secondary changes output atomic accumulation to Float64; private/factor/tensor arithmetic remains Float32. A separate guarded Float64 output is allocated once per mode, shared by every partition dispatch, then projected back to the original Float32 factor buffer. The source restores previous modes to constants; no ALS feedback is implied.

The original strict absolute0.01+relative2e-5 budget remains unchanged. Original Float32 full-size errors reach0.04012859 and fail64 fields; this alone does not establish a native algorithm defect. The named precision intervention passes every full field and independent exact-integer COO oracle: Float64 maximum error0.00038338403, projected Float32 maximum0.00038695354. All20 complete artifacts (4,514,748 Float64 words each, including2,052,307 exact actual-input prefix words) are identical across all four models. Raw original failures and all candidate revisions remain preserved.

## Executed scope

Original toy3D/R32, author140nnz3D/R7, 4692nnzlong-fiber4D/R3 and original Makefile generator dimensions4096×4096×4096×512 with source2% rates, named seed20261020,5,637,372 nonzeros. Every mode and all seven source GPU branches execute (14 launches/pipeline,84 including warmup+five checked trials). All input factor columns are independently checked, not assumed, to have source constant-column values. R>32, arbitrary feature-varying factors, other partition parameters and complete CP iterations remain unverified.

The source CPU terminal-index, initialization and4D copy/extent corrections are preserved separately. Original archives are in reference_history. See secondary_fp64_atomic/intervention_provenance.json and verification_plan.md. No framework unsupported inference follows from the old reference failure.

CUDA.jl owns seven device launches. KA owns seven logical-index kernel dispatches with a CUDA backend atomic extension. JACC owns seven parallel_for entry dispatches with CUDA thread/atomic extensions, explicit zero dynamic shared memory and a separate cached launch specification per(op,grid,block). Full original CPU loading/formatting/controller/checker is shared through the explicit256-byte C ABI. No native GPU fallback is permitted for Julia callbacks.

## Timings and evidence

Five checked in-process trials after full-pipeline warmup; average trials2–5. Means: CUDA6.507031856s; CUDA.jl8.084757395s; KA6.113310944s; JACC7.43556578525s. Preserve all raw timings, including the CUDA.jl late-trial increase. The region includes original CPU loading/formatting/transfers and the new output allocation/initialization/D2H/projection/H2D/free. Added scientific observations are excluded symmetrically; their pauses may affect clocks/caches. These are host-inclusive qualified pipeline measurements, not device-only kernel performance or original Float32 CLI performance. No causal framework ranking or roofline claim follows.

Final receipt: `.porting-state/portable_julia/runs/20261020T084209583131Z_mmcsf/result.json`. Full independently recomputed oracle, guard, read-only, input, projection and source-hash checks: `secondary_fp64_atomic/capture_audit.json`. Actual CSF/index/tensor readbacks are retained once as canonical exact bytes; every model/trial performs actual complete-byte comparisons and reuses that same byte content, rather than claiming that repeated seed/hash equality proves device inputs. Full factors/guarded outputs are independently captured every trial. Original allocations have no invented redzones; only the new Float64 output has explicit64-word guards.

Actual usage boundaries and per-model candidate units are in effort_summary.json. Cached input is a subset of input; reasoning is a subset of output. Ordered mathematical/controller reuse is explicit; tokens are observed inclusive workflow windows, not intrinsic independent framework costs. Missing older consumption is not reconstructed.

## Clean-checkout reproduction

Requires an existing CUDA toolkit/nvcc, Boost headers, Julia1.12.6 and the pinned `_julia_ka_env` / `_julia_jacc_env` projects. No OS installation or privileged command is part of this recipe. Run from the source repository root. Use a fresh `/tmp/mmcsf-reproduction` directory; the fixture generator refuses to overwrite its full input and enforces a300MiB/file limit plus2GiB free reserve.

```sh
python3 src/mmcsf-julia/generate_fixtures.py --output-dir /tmp/mmcsf-reproduction/fixtures --seed 20261020
flock -x /tmp/hecbench-julia-study-gpu.lock env CUDA_VISIBLE_DEVICES=6 python3 src/mmcsf-julia/secondary_fp64_atomic/build_program.py --mode native --output /tmp/mmcsf-reproduction/native
flock -x /tmp/hecbench-julia-study-gpu.lock env CUDA_VISIBLE_DEVICES=6 /tmp/mmcsf-reproduction/native /tmp/mmcsf-reproduction/fixtures /tmp/mmcsf-reproduction/canonical /tmp/mmcsf-reproduction/preparation --prepare
```

Expected full text fixture SHA256 `575e3af43e68294fcb67614b1bf87d921f9eb4f98672e10f5e28c3a53480daa0`,214,010,197bytes. Expected full actual canonical input SHA256 `040e878ed278946408b28065c05470478ecc15a11064be3e980a8bbd185dcd2d`; all fixture/canonical hashes are in fixtures_manifest.json/actual_input_manifest.json. No nonregenerable binary asset is required.

Generate a **new** owned config using `make_config.py --models cuda --fixtures /tmp/mmcsf-reproduction/fixtures --canonical /tmp/mmcsf-reproduction/canonical --output src/mmcsf-julia/secondary_fp64_atomic/study_configs/repro_native.json`, then `python3 src/mmcsf-julia/run_study.py --config PATH`. It prints a canonical result alias. Sequentially generate/run `--models cuda,cuda_jl`, then `cuda,cuda_jl,ka`, then `cuda,cuda_jl,ka,jacc`, each with `--reuse-result PREVIOUS_RESULT` and a distinct config output path. The central verifier owns the GPU lock; **do not outer-lock run_study.py**. `--julia` selects an existing Julia executable; choose nvcc through PATH. New paths/campaigns are new provenance, not a rewriting of archived configs.

Finally run `python3 src/mmcsf-julia/secondary_fp64_atomic/audit_captures.py --result FINAL_RESULT --fixtures /tmp/mmcsf-reproduction/fixtures --output /tmp/mmcsf-reproduction/capture_audit.json` with NumPy available. This recomputes full per-row/per-mode contraction gold using exact signed Float32 integer units, not the GPU/reference values. Cold source generators are optional: the reviewed generated Julia sources are tracked. No additional GPU trial is needed to inspect retained accepted captures.

Large captures live in local `/tmp` with canonical aliases/storage_provenance; preserve them before cleanup/reboot. Raw research is not a compact paper/Git asset.
