# Sparkler full-source-scope four-model study

The original Makefile uses four MPI ranks, 4,000 vectors, 90,000 fields and ten iterations. The source never selects a rank-specific GPU: all four ranks use logical device zero. This study preserves that schedule on physical GPU6; it is **not** a four-GPU scaling experiment. Additional 17/31/3 and 2/1/2 controls exercise uneven rank partitions, padding and ranks with zero true vectors.

The authored GPU initializer generates binary column-major matrices. The original cuBLAS FLOAT32 SGEMM path computes pairwise counts; all logical results are bounded exact FP32 integers. The independent checker uses integer gcd/lcm, separately reconstructs the complete source UInt64 checksum, and checks physical padding and actual guard bytes. UInt64 hashes are serialized as two UInt32 words, never rounded through Float64.

## Evidence

Original CLI diagnostic: `original_native_readiness_receipt.json` (five separate MPI program evaluations, not the primary in-process timing protocol). Full native study: run `20261020T034748612377Z_sparkler`; `native_full_capture_audit.json` independently rehashes 3,040,034,816 complete decompressed CAS bytes. Every measured active SGEMM output is read back, fully checked and hashed. Exact repeats share losslessly compressed CAS storage only after the actual readback/check/hash. Every distinct generated B setup is captured on iteration one, with final B and first/final A captures; repeated internal B generations are executed but are **not** each individually claimed as retained input readbacks.

Full primary artifacts retain all ten active rank/step matrices per case, including padding, and exact source hash/FLOP headers: 40,003,517 UInt32 words per trial. The eleven-word external-argument prefix does not contain generated matrix inputs. Those complete actual inputs have separate physical CAS captures and membership ledgers. Raw CAS and every repeated-call ledger remain local study evidence.

## Timing and interpretation

The primary time sums rank zero's 45 original GEMM/stream-completion/MPI-barrier intervals across the three cases. The authored input kernel, validation, source checksum and physical observers are outside those intervals. Explicit whole-context completion is included symmetrically for the foreign-library stream boundary. Secondary per-rank preparation event times are recorded separately. This is a mixed authored-kernel/library/host-MPI pipeline, with zero authored kernels inside the primary GEMM timer; it cannot isolate framework kernel lowering or multi-GPU scaling. No physical application, general SGEMM conditioning, counter-based roofline or full CLI end-to-end timing claim follows from this integer-pattern witness.

All initial source files remain byte-preserved under `original_native/`. Native Matrix, GPU initializer and `perform_gemm` definitions remain unchanged under guarded allocation instrumentation; unused physical padding is explicitly initialized outside the GEMM ROI. This is not a native mathematical repair and is excluded from untouched original-CLI timing aggregates.

## Reproduction prerequisites

Existing user-space Julia1.12.6/CUDA6.2.1 project environments, CUDA12.0 nvcc, HPCX2.22.1 OpenMPI, pinned system cuBLAS12.0.2, OpenSSL and zlib. No sudo or system-package installation is needed or authorized. The launcher creates four local ranks all on visible device6; only launch through the central verifier so the global GPU lock is held.

```
python3 tools/verify_portable_julia.py --config src/sparkler-julia/study_config_native.json
# Accepted targets reuse immutable references:
python3 tools/verify_portable_julia.py --config src/sparkler-julia/study_config_jacc.json
PYTHONPATH=/tmp/stencil_ports_numpy_226 python3 src/sparkler-julia/audit_capture.py --model cuda --receipt /tmp/sparkler-native-audit.json
```

The NumPy prefix is the existing local installation used here, not a portable installation instruction. `audit_capture.py` accepts `--root` for relocated raw evidence. No external fixture is needed: the source GPU initializer generates all matrices. The full CLI source runtime and preserved raw outputs are not small release assets.

## Final accepted result and effort

Final result: `20261020T040715236989Z_sparkler`; all twenty full artifacts have SHA256 `e756381a39b8ec2c49cae3a1535f26003fe58ce88e43949f2bfa100cc1e50033`. Means2–5: {'cuda': 6.369190231999999, 'cuda_jl': 6.526667549500001, 'ka': 6.378580274000001, 'jacc': 6.3763163115}. `final_source_artifact_audit.json` verifies every bound source and all saved primary hashes. Full per-model capture audit receipts are retained. Native and Julia primary allocation/host-thread binding are instrumented; `--bind-to none` and Julia task-stream reuse versus fresh native per-case streams are explicit covariates.

CUDA.jl had four host-debug failures before one authored-kernel JIT program passed; KA passed its first program; JACC required a public launch-factory repair before the same authored body passed. `candidate_history/` and raw run logs preserve failures. `study_target_backend.cpp` is a diagnostic standard-loader control, not the accepted dependency; accepted targets use unchanged `study_backend.cpp`. See manifests/scope reviews for exact units and calibration qualifications. No source scientific defect or unsupported framework feature was observed.

There are **no external fixture assets to stage**. All small textual code, contracts, audit receipts, source histories and observations form the support packet. Keep generated primary outputs, CAS, shared libraries and executables outside Git. The archived CPU checksum probe has its original absolute include as provenance; use `-I src/sparkler-julia` with a relocated include or compile a temporary adjusted copy when replaying.
