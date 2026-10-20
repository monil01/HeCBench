# Complete leukocyte detector/tracker study

Read [SCOPE.md](SCOPE.md) and `reference_provenance.json` before interpreting results. This is a conditional corrected-native study, not certification of the untouched CUDA CLI or clinical efficacy. All original files are preserved in `original_native/`; corrections and instrumentation are separate directories.

## Preparation

Requires an existing C/C++ compiler, CUDA toolkit supporting sm70, the recorded Julia environment, and sufficient disk space for complete binary evidence. No system package installation or privilege escalation is used.

```
python3 src/leukocyte-julia/prepare.py --archive /path/to/leukocyte.tar.bz
# Alternatively retrieve the exact declared public DVC object:
python3 src/leukocyte-julia/prepare.py --download
```

The archive must be exactly 113303206 bytes, MD5 `79470dcf4adeda242fec899bf0af6553`. The sole extracted AVI must be 184338944 bytes, SHA256 `e3123dd9354293ab41703fc4f9e7c7cf8f7061e09b05d3ad9925941aa19020cc`. The current study recovered and verified an existing local exact archive; the download option implements the declared DVC-v3 public object path and has not been invoked here. The large video/archive are external runtime assets, not Git source artifacts. Generated `.so`, object/archive files, executables, `.bin` oracle extracts, and raw captured outputs are rebuildable evidence/support, not substitute GPU results.

## Execution

The canonical verifier serializes GPU6 with the shared study lock. Do not wrap it in another lock. Configurations use absolute local source paths and need relocation adjustment on other hosts.

```
python3 tools/verify_portable_julia.py --config src/leukocyte-julia/study_config_native.json
python3 tools/verify_portable_julia.py --config src/leukocyte-julia/study_config_cuda_jl.json
```

Native build commands are retained in each run log. `study_config_native_pilot.json` is a one-trial diagnostic pilot, not a five-trial benchmark. The row and one-step control configurations are diagnostic operators, not complete application performance measurements. The primary configuration runs full detection and all original 100 tracking frames plus 1/3-frame controls, five times after a full warmup.

## Independent and raw evidence

`study_full_oracles.cpp` checks all initial GICOV/dilation pixels and selected fixed-count MGVF patch fields. `prepare_full_oracles.py` extracts the original one-trial pilot; `prepare_model_oracles.py` extracts an explicitly selected model's first checked trial. `audit_full_pipeline.py` joins every retained external video/input prefix, GPU stage field, actual guard byte stream, descriptor, and complete serialized numerical artifact. CPU trajectories are complete outputs compared by the central verifier; independent selected-field checking does not establish clinical validity or independent convergence of every trajectory.

Rejected candidates and controls remain under `candidate_history/`, `reference_history/`, raw run directories, and per-model effort/failure logs. In particular, passing local finite/guard checks does not override a complete dual-reference trajectory mismatch. Explicit CPU `libm.so.6` binding and source arithmetic FMA decisions are recorded feedback, not silently corrected historical successes.

## Final accepted receipt

Canonical `study_config_jacc.json`; final run `20261019T232117429587Z_leukocyte` has all four models and all five checked trials. Means2–5 (sum110 kernel regions): CUDA .532646082s, CUDA.jl .862398039s, KA .88439455375s, JACC .86936817425s. All complete Julia artifacts are identical; native maximum difference1.42e-13 is in CPU outputs. JACC uses explicit CUDA static shared scratch and is labeled hybrid. CUDA.jl/KA own framework kernels. All3 share the source-version-compatible pureJulia atan helper and ownJuliaCPU pipeline; no clinicalaccuracyclaim.
