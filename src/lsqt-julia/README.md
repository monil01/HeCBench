# LSQT / GPUQT solver study

Original GPUQT copyright 2017 Zheyong Fan, Ville Vierimaa and Ari Harju; GPL-3.0-or-later. Original notices and full license are preserved under `original_native/`. Julia translations and study instrumentation preserve that attribution.

## Scope

All fourteen native GPU entries and the complete source DOS, VAC0, VAC, MSD, spin-polarization and LDOS host algorithms are exercised. `case_manifest.json` records every parameter/source hash. Two original example models retain their moment counts (1000/500), energies (601/401), random-vector counts (1/2), Anderson disorder W=4 and time schedules (20/30); only lattice sizes are bounded to 16384 sites. Added 12-site lattice, 257-orbital complex Hermitian ring and 18-orbital spin pairs exercise partial blocks and all six solver functions. This is not the original four-million-site whole CLI, arbitrary-input validation, USE_SP/LORENTZ testing, or a transport/stochastic-convergence certificate.

Problem construction is an explicitly shared preinitialized-input stage. `study_fixture_builder.cpp` executes the original CPU model/seeded random-state construction; no scientific result is precomputed into fixtures. Julia implements the actual solver host algorithms and device work, without a C solver fallback. The original fourteen GPU bodies are unchanged by the observer instrumenter (`native_instrumentation_review.json`).

## Inputs and preparation

The exact DVC examples archive is 35,364,096 bytes, MD5 c7d1af7bc30a4faf929c8b399c071024. Current work verified and extracted a preexisting local cache; it did not perform an HTTP download. The content-addressed URL is recorded in `asset_provenance.json`. Generated case inputs and five binary fixtures are small reproducible study assets, distinct from runtime outputs.

Preparation sources: `prepare_cases.py`, `study_fixture_builder.cpp`, `prepare_instrumentation.py`, `study_math_oracle.cpp`. Compiler commands and stdout/stderr are preserved in `readiness_logs/`. `source_provenance.json` binds originals; `case_manifest.json` binds generated problem inputs; `problem_input_audit.json` checks Hermiticity, norms and scaled spectral bounds.

## Numerical and physical evidence

Predeclared primary tolerance is 1e-10 absolute + 1e-8 relative. Every checked trial includes complete full-precision scientific values, complex moments before damping, final solver states and every released device allocation. The serialized exact input prefix retains actual Hamiltonian buffers, every initial state and scalar Bessel/operator arguments. Separate raw streams preserve physical device buffers, complete 256-byte prefix/suffix guards and lifetime/layout joins. Ordinary six-digit output is also retained but is not the numerical oracle.

`audit_mathematical_outputs.py` compares the source outputs to independent long-double dense/CSR Chebyshev witnesses and 96-term Taylor matrix exponential/derivative controls; the original source uses Bessel/Chebyshev propagation. Large initial DOS moments and appropriate final-state norms are checked independently. `Bessel_truncation_scope_review.json` derives an ideal Hermitian-operator tail bound, separate from floating-point/stochastic error. Wrapped bond displacements define the periodic current operator; they are not a global diagonal-X commutator at the boundary. Source spin/LDOS normalization is retained and qualified.

## Timing and verification

Use `python3 tools/verify_portable_julia.py --config src/lsqt-julia/study_config_MODEL.json` from the repository root (GPU6, global study lock acquired by the runner). Each process warms all five cases, executes five checked trials and averages timings 2–5. Timed regions sum the original run_X solver-function boundaries, including host damping/Bessel/energy sums, allocations/destruction/transfers, GPU work and normal text output. Input model construction and raw observer file capture are outside the region. Guard allocation enlargement, explicit completion fences and scalar/counter bookkeeping are disclosed measurement covariates. These numbers are not pure device-kernel throughput or unchanged whole-CLI wall time, and are excluded from untouched-native aggregate timings.

Raw captures, compiler executables, oracle executables and cached archive/extraction are local research outputs, not source assets. Audits rehash raw bytes and join all release payloads and guards to their trial artifacts. Preserve all rejected diagnostic runs. Do not claim literal host/device pointer identity across processes.

## Final result and regeneration

All four models pass: `all4_final_receipt.json` and `final_source_binding_audit.json`; canonical configuration `study_config_jacc.json`. Independent audits are `*_mathematical_audit.json` / `*_raw_audit.json`, with native audits prefixed `native_primary_`. All three Julia primary artifacts are bit-identical; native agreement uses the declared componentwise tolerance.

From the repository root, regenerate the small input fixtures and independent CPU oracles in a separate directory:

```sh
python3 src/lsqt-julia/prepare_runtime_assets.py --output-dir /tmp/lsqt-recreated --cxx g++
```

The generator prints concrete commands and SHA comparisons against the recorded assets. This CPU-only replay was added after verification as a portable wrapper around the commands actually used in `readiness_logs/`; it does not replace or retime any measured run. It needs existing C++17 compiler/libm and no privileged installation. `prepare_instrumentation.py` recreates the observer copy from preserved originals; do not rerun generators over frozen measured sources while verification is active. Julia launch-entry generator paths record this checkout and are provenance, not needed to execute the committed entrypoints.

All actual token windows, count units and scientific limitations appear in each child `translation_accounting.json`, `paper_observations.json` and the local STATE manifests. Original native preparation is separately measured and not added to target translation windows; overlapping verification windows are not additive.
